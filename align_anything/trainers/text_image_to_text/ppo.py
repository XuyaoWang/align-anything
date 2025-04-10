# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trainer for PPO training."""


import argparse
import copy
import os
import sys
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
from transformers import GenerationConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_image_to_text import (
    PromptOnlyBatch,
    PromptOnlyDataset,
    SupervisedDataset,
)
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.trainers.text_to_text.ppo import PPOTrainer as PPOTextTrainer
from align_anything.utils.device_utils import torch_set_device
from align_anything.utils.multi_process import (
    get_all_reduce_max,
    get_all_reduce_mean,
    get_current_device,
)
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    gather_log_probabilities,
    is_same_tokenizer,
    masked_mean,
    read_cfgs,
    remove_pad_tokens,
    seed_everything,
    update_dict,
)


def move_padding_left(input_tensor, padding_value=0):
    """Moves the padding values in each row of the input_tensor from the right to the left.

    Args:
        input_tensor (Tensor): A 2D tensor to be processed.
        padding_value (int): The value used for padding, default is 0.

    Returns:
        Tensor: The tensor with padding values moved to the left.
    """
    # Calculate the number of padding elements at the start of each row
    start_pad_counts = (
        (input_tensor == padding_value)
        .cumsum(dim=1)
        .eq(torch.arange(1, input_tensor.size(1) + 1, device=input_tensor.device))
        .sum(dim=1)
    )
    # Calculate the number of non-padding elements in each row
    non_pad_counts = (input_tensor != padding_value).sum(dim=1)
    # Create a new tensor of the same size as input_tensor, filled with padding_value
    output_tensor = torch.full_like(input_tensor, padding_value, device=input_tensor.device)
    # Get the indices for each row
    max_len = input_tensor.size(1)
    indices = torch.arange(max_len, device=input_tensor.device).expand(len(non_pad_counts), max_len)
    # Calculate the shift for each row
    shifts = max_len - non_pad_counts.unsqueeze(1) - start_pad_counts.unsqueeze(1)
    # Compute the new indices
    new_indices = (indices - shifts) % max_len
    # Rearrange the tensor using the gather function
    output_tensor = torch.gather(input_tensor, 1, new_indices)

    return output_tensor


class PPOTrainer(PPOTextTrainer):  # pylint: disable=too-many-instance-attributes
    """Trainer base class for PPO training."""

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        # load training datasets
        self.prompt_only_dataloader, self.eval_dataloader, self.ptx_dataloader = (
            self.get_dataloaders(PromptOnlyDataset, PromptOnlyDataset, SupervisedDataset)
        )

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_cfgs)
        if self.ds_eval_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf_eval = HfDeepSpeedConfig(self.ds_eval_cfgs)
        # loading actor model
        self.actor_model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            freeze_mm_proj=self.cfgs.train_cfgs.freeze_mm_proj,
            freeze_vision_tower=self.cfgs.train_cfgs.freeze_vision_tower,
            freeze_language_model=self.cfgs.train_cfgs.freeze_language_model,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        self.tokenizer.model_max_length = self.cfgs.model_cfgs.model_max_length
        # loading actor reference model
        self.actor_reference_model, _, _ = load_pretrained_models(
            self.cfgs.model_cfgs.actor_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # loading reward model
        self.reward_model, self.reward_tokenizer, _ = load_pretrained_models(
            self.cfgs.model_cfgs.reward_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            is_reward_model=True,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # loading reward critic model
        self.reward_critic_model, self.reward_critic_tokenizer, _ = load_pretrained_models(
            self.cfgs.model_cfgs.reward_critic_model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            is_reward_model=True,
            freeze_mm_proj=self.cfgs.train_cfgs.freeze_mm_proj,
            freeze_vision_tower=self.cfgs.train_cfgs.freeze_vision_tower,
            freeze_language_model=self.cfgs.train_cfgs.freeze_language_model,
            processor_kwargs=self.cfgs.train_cfgs.processor_kwargs,
        )
        # initial checking
        if is_same_tokenizer(self.tokenizer, self.reward_tokenizer):
            self.reward_tokenizer = self.tokenizer
        if not is_same_tokenizer(self.tokenizer, self.reward_critic_tokenizer):
            raise ValueError(
                (
                    'Reward critic tokenizer must be the same as actor tokenizer. '
                    'Expected {0.__module__}.{0.__qualname__}(vocab_size={1}), '
                    'but got {2.__module__}.{2.__qualname__}(vocab_size={3}). '
                    'Please consider pass `--reward_critic_model_name_or_path` from the command line.'
                ).format(
                    type(self.tokenizer),
                    len(self.tokenizer),
                    type(self.reward_critic_tokenizer),
                    len(self.reward_critic_tokenizer),
                ),
            )

        # training setup
        self.generation_config = GenerationConfig(
            max_new_tokens=self.cfgs.model_cfgs.max_new_tokens,
            temperature=self.cfgs.model_cfgs.temperature,
            top_p=self.cfgs.model_cfgs.top_p,
            repetition_penalty=self.cfgs.model_cfgs.repetition_penalty,
            do_sample=True,
        )

    def actor_step(
        self, mini_prompt_only_batch: PromptOnlyBatch
    ) -> list[dict[str, Any], list[int]]:
        infer_batch = self.infer_batch(mini_prompt_only_batch)
        actor_batch = copy.deepcopy(infer_batch)
        sequences = self.actor_model.module.generate(
            **infer_batch,
            generation_config=self.generation_config,
            synced_gpus=True,
            do_sample=True,
        )
        sequences = move_padding_left(sequences.contiguous(), self.tokenizer.pad_token_id)
        attention_mask = sequences.not_equal(self.tokenizer.pad_token_id)
        actor_batch['input_ids'] = sequences
        actor_batch['attention_mask'] = attention_mask

        response_lens = []
        batch_size = sequences.size(0)
        for idx in range(batch_size):
            prompt_length = len(
                remove_pad_tokens(
                    mini_prompt_only_batch['input_ids'][idx].squeeze().tolist(),
                    self.tokenizer.pad_token_id,
                )
            )
            sequence_wo_pad = remove_pad_tokens(
                sequences[idx].squeeze().tolist(), self.tokenizer.pad_token_id
            )
            response = sequence_wo_pad[prompt_length:]
            response_lens.append(len(response))
        return actor_batch, response_lens

    @torch.no_grad()
    def rollout(self, prompt_only_batch: PromptOnlyBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""
        # freeze the model for rolling out
        self.set_train(mode=False)

        micro_inference_batches = []
        micro_training_batches = []
        mini_batch = prompt_only_batch.copy()
        # actor generation
        actor_batch, response_lens = self.actor_step(mini_batch)
        # reward model and reward critic model scoring
        reward_batch = self.reward_model_step(actor_batch)
        # calculate the log probabilities
        logits = self.actor_model(**actor_batch).logits
        ref_logits = self.actor_reference_model(**actor_batch).logits

        logprob_list = []
        ref_logprob_list = []
        reward_value_list = []

        batch_size = logits.size(0)

        for idx in range(batch_size):
            response_length = response_lens[idx]
            input_id = actor_batch['input_ids'][idx, 1:][-response_length:].unsqueeze(0)

            logit = logits[idx, :-1][-response_length:].unsqueeze(0)
            ref_logit = ref_logits[idx, :-1][-response_length:].unsqueeze(0)
            reward_value = reward_batch['reward_values'][idx][-response_length:].unsqueeze(0)

            logprob_list.append(gather_log_probabilities(logit, input_id).squeeze())
            ref_logprob_list.append(gather_log_probabilities(ref_logit, input_id).squeeze())
            reward_value_list.append(reward_value.squeeze())

        log_probs = torch.nn.utils.rnn.pad_sequence(
            logprob_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        ref_log_probs = torch.nn.utils.rnn.pad_sequence(
            ref_logprob_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        reward_values = torch.nn.utils.rnn.pad_sequence(
            reward_value_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        response_mask = (log_probs != 0).bool().to(logits.device)

        micro_training_batch = {}
        micro_training_batch['response_lens'] = response_lens
        micro_training_batch['log_probs'] = log_probs
        micro_training_batch['ref_log_probs'] = ref_log_probs
        micro_training_batch['reward'] = reward_batch['reward']
        micro_training_batch['reward_values'] = reward_values
        micro_training_batch['response_mask'] = response_mask

        mini_batch['input_ids'] = reward_batch['input_ids']
        mini_batch['attention_mask'] = actor_batch['attention_mask']
        # add rollout results to the batches
        micro_inference_batches.append(mini_batch)
        micro_training_batches.append(micro_training_batch)

        # unfreeze the model for training
        self.set_train()

        return micro_inference_batches, micro_training_batches

    def rl_step(
        self, inference_batch: dict[str, torch.Tensor], training_batch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Perform a single update step with RL loss."""
        response_lens = training_batch['response_lens']
        old_log_probs = training_batch['log_probs']
        ref_log_probs = training_batch['ref_log_probs']
        reward = training_batch['reward']
        old_reward_values = training_batch['reward_values']

        input_ids = inference_batch['input_ids']
        batch_size = input_ids.size(0)
        sequence_mask = training_batch['response_mask']

        with torch.no_grad():
            old_rewards = self.add_kl_divergence_regularization(
                reward,
                old_log_probs,
                ref_log_probs,
                sequence_mask,
            )
            reward_advantages, reward_returns = self.get_advantages_and_returns(
                old_reward_values,
                old_rewards,
                sequence_mask,
                start=0,
            )
        logits = self.actor_model(**self.infer_batch(inference_batch), use_cache=False).logits
        logprob_list = []

        for idx in range(batch_size):
            response_length = response_lens[idx]
            input_id = input_ids[idx, 1:][-response_length:].unsqueeze(0)
            logit = logits[idx, :-1][-response_length:].unsqueeze(0)
            logprob_list.append(gather_log_probabilities(logit, input_id).squeeze())

        log_probs = torch.nn.utils.rnn.pad_sequence(
            logprob_list, batch_first=True, padding_value=0.0
        ).to(logits.device)
        actor_loss = self.actor_loss_fn(
            log_probs,
            old_log_probs,
            reward_advantages,
            sequence_mask,
        )
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        raw_reward_values = self.reward_critic_model(**self.infer_batch(inference_batch)).scores
        raw_reward_values = raw_reward_values.squeeze(dim=-1)[:, :-1]

        reward_value_list = []

        for idx in range(batch_size):
            response_length = response_lens[idx]
            reward_value = raw_reward_values[idx][-response_length:].unsqueeze(0)
            reward_value_list.append(reward_value.squeeze())
        reward_values = torch.nn.utils.rnn.pad_sequence(
            reward_value_list, batch_first=True, padding_value=0.0
        ).to(logits.device)

        reward_critic_loss = self.critic_loss_fn(
            reward_values,
            old_reward_values,
            reward_returns,
            sequence_mask,
        )
        self.reward_critic_model.backward(reward_critic_loss)
        self.reward_critic_model.step()

        with torch.no_grad():
            mask = sequence_mask
            kl_divergence = ((old_log_probs - ref_log_probs) * mask).sum(dim=-1).mean()
            mean_generated_length = mask.sum(dim=-1).float().mean()
            max_generated_length = mask.sum(dim=-1).float().max()

            reward = reward.mean()
            reward_with_kl_penalty = (old_rewards * mask).sum(dim=-1).mean()
            reward_advantage = masked_mean(reward_advantages, mask)
            reward_return = masked_mean(reward_returns, mask)
            reward_value = masked_mean(reward_values, mask)

            actor_loss = get_all_reduce_mean(actor_loss)
            reward_critic_loss = get_all_reduce_mean(reward_critic_loss)
            reward = get_all_reduce_mean(reward)
            reward_with_kl_penalty = get_all_reduce_mean(reward_with_kl_penalty)
            reward_advantage = get_all_reduce_mean(reward_advantage)
            reward_return = get_all_reduce_mean(reward_return)
            reward_value = get_all_reduce_mean(reward_value)
            kl_divergence = get_all_reduce_mean(kl_divergence)
            mean_generated_length = get_all_reduce_mean(mean_generated_length)
            max_generated_length = get_all_reduce_max(max_generated_length)

        dist.barrier()

        return {
            'train/actor_loss': actor_loss.item(),
            'train/reward_critic_loss': reward_critic_loss.item(),
            'train/reward': reward.item(),
            'train/reward_with_kl_penalty': reward_with_kl_penalty.item(),
            'train/reward_advantage': reward_advantage.item(),
            'train/reward_return': reward_return.item(),
            'train/reward_value': reward_value.item(),
            'train/kl_divergence': kl_divergence.item(),
            'train/actor_lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/reward_critic_lr': self.reward_critic_model.optimizer.param_groups[0]['lr'],
            'train/mean_generated_length': mean_generated_length.item(),
            'train/max_generated_length': max_generated_length.item(),
        }

    def add_kl_divergence_regularization(
        self,
        reward: torch.Tensor,  # size = (B,)
        log_probs: torch.Tensor,  # size = (B, L)
        ref_log_probs: torch.Tensor,  # size = (B, L)
        sequence_mask: torch.BoolTensor,  # size = (B, L)
    ) -> torch.Tensor:  # size = (B, L)
        """Add KL divergence regularization on scalar rewards."""
        end_index = torch.cat([m.nonzero()[-1] for m in sequence_mask])  # size = (B,)

        # size = (B, L)
        kl_divergence_estimate = log_probs - ref_log_probs
        kl_penalty_rewards = -self.kl_coeff * kl_divergence_estimate
        rewards = torch.scatter_add(
            kl_penalty_rewards,
            dim=-1,
            index=end_index.unsqueeze(dim=-1),
            src=reward.to(kl_penalty_rewards.dtype).unsqueeze(dim=-1),
        )
        return torch.clamp(rewards, min=-self.clip_range_score, max=self.clip_range_score)

    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        sequence_mask: torch.BoolTensor,
        start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using Generalized Advantage Estimation (GAE)."""
        # Modified from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
        last_gae_lambda = 0.0
        advantages_reversed = []
        values = values * sequence_mask
        rewards = rewards * sequence_mask
        length = rewards.size(-1)
        for t in reversed(range(start, length)):  # pylint: disable=invalid-name
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
            advantages_reversed.append(last_gae_lambda)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch_set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_image_to_text', 'ppo')
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)

    # get custom configs from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # finetune the model
    trainer = PPOTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
