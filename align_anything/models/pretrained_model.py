# Copyright 2024 PKU-Alignment Team and tatsu-lab. All Rights Reserved.
#
# This code is inspired by the tatsu-lab's stanford-alpaca library.
# https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
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

from __future__ import annotations

import contextlib
import os
import warnings
from typing import Any, Callable, Literal

import deepspeed
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, AutoModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from align_anything.models.model_registry import AnyModel
from align_anything.utils.multi_process import is_main_process


DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'


# Reference: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
def resize_tokenizer_embedding(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> None:
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    def verify_vocabulary_embedding_sizes(
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        format_message: Callable[[Any, Any], str],
    ) -> None:
        input_embeddings = model.get_input_embeddings()
        if (
            input_embeddings is not None
            and input_embeddings.num_embeddings != len(tokenizer)
            and is_main_process()
        ):
            warnings.warn(
                format_message(len(tokenizer), input_embeddings.num_embeddings),
                category=RuntimeWarning,
                stacklevel=3,
            )

    def init_new_embeddings(
        embeddings: nn.Embedding | nn.Linear | None,
        new_num_embeddings: int,
        num_new_embeddings: int,
    ) -> None:
        if embeddings is None:
            return

        params = [embeddings.weight, getattr(embeddings, 'bias', None)]
        context = (
            deepspeed.zero.GatheredParameters(params, modifier_rank=0)
            if is_deepspeed_zero3_enabled()
            else contextlib.nullcontext()
        )
        with context:
            for param in params:
                if param is None:
                    continue
                assert param.size(0) == new_num_embeddings
                param_data = param.data
                param_mean = param_data[:-num_new_embeddings].mean(dim=0, keepdim=True)
                param_data[-num_new_embeddings:] = param_mean

    verify_vocabulary_embedding_sizes(
        tokenizer=tokenizer,
        model=model,
        format_message=(
            'The tokenizer vocabulary size ({}) is different from '
            'the model embedding size ({}) before resizing.'
        ).format,
    )

    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    new_num_embeddings = len(tokenizer)

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if num_new_tokens > 0:
        hf_device_map = getattr(model, 'hf_device_map', {})
        devices = {
            torch.device(device)
            for device in hf_device_map.values()
            if device not in {'cpu', 'disk'}
        }
        is_model_parallel = len(devices) > 1

        if not is_model_parallel:
            model.resize_token_embeddings(new_num_embeddings)
            init_new_embeddings(
                model.get_input_embeddings(),
                new_num_embeddings=new_num_embeddings,
                num_new_embeddings=num_new_tokens,
            )
            init_new_embeddings(
                model.get_output_embeddings(),
                new_num_embeddings=new_num_embeddings,
                num_new_embeddings=num_new_tokens,
            )

    verify_vocabulary_embedding_sizes(
        tokenizer=tokenizer,
        model=model,
        format_message=(
            'The tokenizer vocabulary size ({}) is different from '
            'the model embedding size ({}) after resizing.'
        ).format,
    )


def load_pretrained_models(  # pylint: disable=too-many-arguments
    model_name_or_path: str | os.PathLike,
    model_max_length: int = 512,
    padding_side: Literal['left', 'right'] = 'right',
    auto_device_mapping: bool = False,
    freeze_vision_tower: bool = True,
    freeze_mm_proj: bool = True,
    dtype: torch.dtype | str | None = 'auto',
    *,
    cache_dir: str | os.PathLike | None = None,
    trust_remote_code: bool = False,
    auto_model_args: tuple[Any, ...] = (),
    auto_model_kwargs: dict[str, Any] | None = None,
    auto_tokenizer_args: tuple[Any, ...] = (),
    auto_tokenizer_kwargs: dict[str, Any] | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin]:
    """Load pre-trained model and tokenizer from a given path."""
    model_name_or_path = os.path.expanduser(model_name_or_path)
    cache_dir = os.path.expanduser(cache_dir) if cache_dir is not None else None
    device_map = 'auto' if auto_device_mapping else None
    if auto_model_kwargs is None:
        auto_model_kwargs = {}
    if auto_tokenizer_kwargs is None:
        auto_tokenizer_kwargs = {}

    model = AnyModel.from_pretrained(
        model_name_or_path,
        *auto_model_args,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        **auto_model_kwargs,
    )

    forbidden_modules = set()
    if freeze_vision_tower:
        forbidden_modules.add('vision_tower')
    if freeze_mm_proj:
        forbidden_modules.add('multi_modal_projector')
    for name, param in model.named_parameters():
        if not any(forbidden_module in name for forbidden_module in forbidden_modules):
            if dtype == torch.float32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        *auto_tokenizer_args,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side=padding_side,
        trust_remote_code=trust_remote_code,
        **auto_tokenizer_kwargs,
    )

    try:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        resize_tokenizer_embedding(tokenizer=processor.tokenizer, model=model)
    except:
        processor = None
    return model, tokenizer, processor
