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

from __future__ import annotations
import sys
from typing import Any, Callable, List
from typing_extensions import TypedDict  # Python 3.10+

import torch
import transformers
from torch.utils.data import Dataset
from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

from align_anything.models.any_model import ModalityType
from align_anything.utils.multi_process import get_current_device
from align_anything.utils.template_registry import get_template_class
from align_anything.utils.tools import right_padding
from datasets import load_dataset, DatasetDict

__all__ = [
    'MMPreferenceDataset',
    'MMPreferenceCollator',
    'MMPreferenceSample',
    'MMPreferenceBatch',
]


class MMPreferenceSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.LongTensor  # size = (L,)
    better_pixel_values: torch.FloatTensor | None  # size = (B, C, H, W) or (B, T, C, H, W) or (B, N, C, M, F)
    worse_pixel_values: torch.FloatTensor | None  # size = (B, C, H, W) or (B, T, C, H, W) or (B, N, C, M, F)
    modality: ModalityType | None

class MMPreferenceBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    labels: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    pixel_values: torch.FloatTensor | None  # size = (B, C, H, W) or (B, T, C, H, W) or (B, N, C, M, F)
    modality: List[ModalityType] | None

# Multi modality preference dataset
class MMPreferenceDataset(Dataset):
    def __init__(
        self,
        path: str,
        template: str,
        tokenizer: transformers.PreTrainedTokenizer,
        processor: transformers.ProcessorMixin | None = None,
        size: int | None = None,
        split: str | None = None,
        subset: str | None = None,
        data_files: str | None = None,
    ):
        super().__init__()
        assert path, f'You must set the valid datasets path! Here is {path}'
        assert template, f'You must set the valid template path! Here is {template}'
        self.tokenizer = processor.tokenizer
        self.processor = processor
        self.raw_data = load_dataset(path, split=split, subset=subset, data_files=data_files)
        if size:
            size = min(size, len(self.raw_data))
            self.raw_data = self.raw_data.select(range(int(size)))
        self.template = get_template_class(template)

    def preprocess(self, raw_sample: dict[str, Any]) -> MMPreferenceSample:
        formatted_sample = self.template.format_sample(raw_sample)
        return_dict = {}

        raw_prompt = ''

        if isinstance(formatted_sample['prompt'], list):
            raw_prompt = self.tokenizer.eos_token.join(formatted_sample['prompt'])
        elif isinstance(formatted_sample['prompt'], str):
            raw_prompt = formatted_sample['prompt'] + self.tokenizer.eos_token
        else:
            raise NotImplementedError
        return_dict['input_ids'] = self.tokenize(raw_prompt)

        return_dict['better_pixel_values'] = self.processor(data_paths=raw_sample['better_data_path'], modality=raw_sample['modality'])['pixel_values']
        return_dict['worse_pixel_values'] = self.processor(data_paths=raw_sample['worse_data_path'], modality=raw_sample['modality'])['pixel_values']
        return_dict['modality'] = raw_sample['modality']

        return return_dict

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return MMPreferenceCollator(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool | str | PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation: bool | str | TruncationStrategy = TruncationStrategy.LONGEST_FIRST,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenize a text string into a tensor representation."""
        if max_length is None:

            max_length = min(self.tokenizer.model_max_length, sys.maxsize)

        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors='pt',
        )['input_ids'][0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        raw_sample = self.raw_data[index]
        data = self.preprocess(raw_sample)
        return data

    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.raw_data)


class MMPreferenceCollator:

    def __init__(self, pad_token_id: int) -> None:
        """Initialize a collator."""
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[MMPreferenceSample]) -> tuple[MMPreferenceBatch]:
        return_dict = {}
        current_device = get_current_device()

        input_ids = [sample['input_ids'] for sample in samples] + [
            sample['input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        return_dict['input_ids'] = right_padding(input_ids, padding_value=self.pad_token_id).to(
            current_device
        )  # size = (2 * B, L)

        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)
        return_dict['attention_mask'] = right_padding(attention_mask, padding_value=0).to(
            current_device
        )  # size = (2 * B, L)

        better_pixel_values = torch.stack([sample['better_pixel_values'] for sample in samples])
        worse_pixel_values = torch.stack([sample['worse_pixel_values'] for sample in samples])
        stacked_pixel_values = torch.cat([better_pixel_values, worse_pixel_values], dim=0)
        return_dict['pixel_values'] = stacked_pixel_values.to(current_device).to(torch.bfloat16)  # size = (2 * B, C, H, W) or (2 * B, T, C, H, W) or (2 * B, N, C, M, F)

        return_dict['modality'] = [sample['modality'] for sample in samples]

        return return_dict
