import os
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import AutoConfig, AutoTokenizer, AutoProcessor, AutoModel, AutoModelForCausalLM
from transformers.models.auto import CONFIG_MAPPING
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.processing_utils import ProcessorMixin
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.feature_extraction_utils import BatchFeature
from transformers.tokenization_utils_base import (
    TextInput,
    TensorType,
    PaddingStrategy,
    PreTokenizedInput,
    TruncationStrategy
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from align_anything.utils.processor_mm import (
    load_and_transform_image_data,
    load_and_transform_video_data,
    load_and_transform_audio_data
)
from align_anything.models.ImageBind import *

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    VISION = "vision" # For Imagebind

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, ModalityType):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False

    def __hash__(self):
        return hash(self.value)

_CONFIG_FOR_DOC = "AnyModelConfig"

class AnyModelConfig(PretrainedConfig):
    model_type = "any_model"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        modality_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=128256,
        video_token_index=128257,
        audio_token_index=128258,
        projector_hidden_act="gelu",
        **kwargs,
    ):

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.modality_config = modality_config
        self.text_config = text_config
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.audio_token_index = audio_token_index
        self.projector_hidden_act = projector_hidden_act

        super().__init__(
            **kwargs,
        )

class AnyModelProcessor(ProcessorMixin):
    # TODO: Add support for any_model_processor
    # attributes = ["any_model_processor", "tokenizer"]
    attributes = ["tokenizer"]
    valid_kwargs = ["chat_template"]
    any_model_processor_class = "AnyModelProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(tokenizer, **kwargs)
        if self.tokenizer is not None:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "<video>", "<audio>"]})

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        data_paths: Union[str, List[str]] = None,
        modality: Optional[Union[ModalityType, List[ModalityType]]] = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:

        if data_paths is not None:
            if modality is None:
                raise ValueError("modality must be specified when data_paths is provided")
            if isinstance(modality, list):
                assert len(set(modality)) == 1, "only one kind modality can be provided in a batch"
                modality = modality[0]

            proceesor_func = None
            if modality == ModalityType.IMAGE:
                proceesor_func = load_and_transform_image_data
            elif modality == ModalityType.VIDEO:
                proceesor_func = load_and_transform_video_data
            elif modality == ModalityType.AUDIO:
                proceesor_func = load_and_transform_audio_data
            else:
                raise ValueError("modality must be one of ModalityType.IMAGE, ModalityType.VIDEO, ModalityType.AUDIO")

            if isinstance(data_paths, str):
                pixel_values = proceesor_func(data_paths)
            else:
                pixel_values = torch.stack([proceesor_func(data_path) for data_path in data_paths], dim=0)
        else:
            pixel_values = None
        if text is None:
            text_inputs = {}
        else:
            text_inputs = self.tokenizer(
                text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
            )

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values, "modality": modality})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_class_input_names = self.feature_extractor_class.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_class_input_names))

@dataclass
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->AnyModel
class AnyModelCausalLMOutputWithPast(ModelOutput):
    """
    Base class for AnyModel causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        modality_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            modality_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    modality_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    modality: Optional[ModalityType] = None


class AnyModelMultiModalProjector(nn.Module):
    def __init__(self, config: AnyModelConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.modality_config["hidden_size"], config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, modality_features):
        hidden_states = self.linear_1(modality_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class AnyModelPreTrainedModel(PreTrainedModel):
    config_class = AnyModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AnyModelAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(self, config: AnyModelConfig):
        self.config = config
        super().__init__(config)


    def _init_weights(self, module):
        # important: this ported version of AnyModel isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa

ANYMODEL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details ([]`AnyModelProcessor`] uses
            [`CLIPImageProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class AnyModelForConditionalGeneration(AnyModelPreTrainedModel):
    def __init__(self, config: AnyModelConfig):
        super().__init__(config)

        self.image_projector = AnyModelMultiModalProjector(config)
        self.video_projector = AnyModelMultiModalProjector(config)
        self.audio_projector = AnyModelMultiModalProjector(config)
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.modality_tower, _ = \
            imagebind_model.imagebind_huge(pretrained=True, store_path=os.path.join(config._name_or_path, config.modality_config["imagebind_ckpt_path"]))
        self.modality_tower = self.modality_tower.to(self.language_model.device)
        self.modality_tower = self.modality_tower.to(self.language_model.dtype)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def _merge_input_ids_with_video_features(self, video_features, inputs_embeds, input_ids, attention_mask, labels):
        num_videos, num_video_patches, embed_dim = video_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special video tokens are
        special_video_token_mask = input_ids == self.config.video_token_index
        num_special_video_tokens = torch.sum(special_video_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_video_tokens.max() * (num_video_patches - 1)) + sequence_length
        batch_indices, non_video_indices = torch.where(input_ids != self.config.video_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged video-text sequence.
        # `special_video_token_mask` identifies video tokens. Each video token will be replaced by `nb_text_tokens_per_videos - 1` text tokens.
        # `torch.cumsum` computes how each video token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_video_token_mask * (num_video_patches - 1) + 1), -1) - 1
        nb_video_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_video_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_video_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_video_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_video_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<video>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the video features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_video_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_video_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_video_indices]

        # 5. Fill the embeddings corresponding to the videos. Anything that is not `text_positions` needs filling (#29835)
        video_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        video_to_overwrite[batch_indices, text_to_overwrite] = False
        video_to_overwrite &= video_to_overwrite.cumsum(-1) - 1 >= nb_video_pad[:, None].to(target_device)

        if video_to_overwrite.sum() != video_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of video tokens is {torch.sum(special_video_token_mask)} while"
                f" the number of video given to the model is {num_videos}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[video_to_overwrite] = video_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= video_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def _merge_input_ids_with_audio_features(self, audio_features, inputs_embeds, input_ids, attention_mask, labels):
        num_audios, num_audio_patches, embed_dim = audio_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.config.audio_token_index
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_audio_tokens.max() * (num_audio_patches - 1)) + sequence_length
        batch_indices, non_audio_indices = torch.where(input_ids != self.config.audio_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `nb_text_tokens_per_audios - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_audio_token_mask * (num_audio_patches - 1) + 1), -1) - 1
        nb_audio_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]

        # 5. Fill the embeddings corresponding to the audios. Anything that is not `text_positions` needs filling (#29835)
        audio_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        audio_to_overwrite &= audio_to_overwrite.cumsum(-1) - 1 >= nb_audio_pad[:, None].to(target_device)

        if audio_to_overwrite.sum() != audio_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {torch.sum(special_audio_token_mask)} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    @add_start_docstrings_to_model_forward(ANYMODEL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AnyModelCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        modality: Optional[ModalityType] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AnyModelCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AnyModelForConditionalGeneration

        >>> model = AnyModelForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                assert modality is not None, "modality must be provided when pixel_values is not None"
                if isinstance(modality, list):
                    assert len(set(modality)) == 1, "only one kind modality can be provided in a batch"
                    modality = modality[0]

                if modality == ModalityType.IMAGE:
                    modality_outputs = self.modality_tower({
                        str(ModalityType.VISION): pixel_values
                    })[str(ModalityType.VISION)] # size = (b, h)
                    image_features = self.image_projector(modality_outputs).unsqueeze(1) # size = (b, 1, h)
                    inputs_embeds = inputs_embeds.to(image_features.dtype)
                    inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                        image_features, inputs_embeds, input_ids, attention_mask, labels
                    )
                elif modality == ModalityType.VIDEO:
                    modality_outputs = self.modality_tower({
                        str(ModalityType.VISION): pixel_values
                    })[str(ModalityType.VISION)] # size = (b, h)
                    video_features = self.video_projector(modality_outputs).unsqueeze(1) # size = (b, 1, h)
                    inputs_embeds = inputs_embeds.to(video_features.dtype)
                    inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_video_features(
                        video_features, inputs_embeds, input_ids, attention_mask, labels
                    )
                elif modality == ModalityType.AUDIO:
                    modality_outputs = self.modality_tower({
                        str(ModalityType.AUDIO): pixel_values
                    })[str(ModalityType.AUDIO)] # size = (b, h)
                    audio_features = self.audio_projector(modality_outputs).unsqueeze(1) # size = (b, 1, h)
                    inputs_embeds = inputs_embeds.to(audio_features.dtype)
                    inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_audio_features(
                        audio_features,
                        inputs_embeds, input_ids, attention_mask, labels
                    )
                else:
                    raise ValueError(f"modality {modality} is not supported")

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return AnyModelCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )
        return model_inputs

    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)



AutoConfig.register("any_model", AnyModelConfig)
AutoProcessor.register(AnyModelConfig, AnyModelProcessor)
AutoTokenizer.register(AnyModelConfig, AnyModelProcessor)
AutoModel.register(AnyModelConfig, AnyModelPreTrainedModel)
AutoModel.register(AnyModelConfig, AnyModelForConditionalGeneration)

@dataclass
class AccustomedAnyModelOutput(AnyModelCausalLMOutputWithPast):
    """
    Base class for AnyModel causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    modality_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # TODO: Change to modality to overwrite
    image_to_overwrite: Optional[torch.BoolTensor] = None
    modality: Optional[ModalityType] = None


class AccustomedAnyModel(AnyModelForConditionalGeneration):

    @classmethod
    def pretrain_class(cls) -> AnyModelPreTrainedModel:
        return AnyModelPreTrainedModel

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, attention_mask, labels
    ):
        return super()._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, labels)

    def _merge_input_ids_with_video_features(
        self, video_features, inputs_embeds, input_ids, attention_mask, labels
    ):
        return super()._merge_input_ids_with_video_features(video_features, inputs_embeds, input_ids, attention_mask, labels)

    def _merge_input_ids_with_audio_features(
        self, audio_features, inputs_embeds, input_ids, attention_mask, labels
    ):
        return super()._merge_input_ids_with_audio_features(audio_features, inputs_embeds, input_ids, attention_mask, labels)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        modality=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> AccustomedAnyModelOutput:
        output = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            modality=modality,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # TODO: change to X_to_overwrite for PPO
        return AccustomedAnyModelOutput(
            loss=output.loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            modality_hidden_states=output.modality_hidden_states,
            image_to_overwrite=None,
            modality=modality
        )
