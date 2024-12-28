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

import json
import re

import librosa
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from datasets import load_dataset


USER_PROMPT_TEMPLATE = """
Based on the audio you hear, answer the following multiple-choice questions by providing only the corresponding option (A, B, C, or D).
Question: {question}
Choice: {choice}
"""

processor = AutoProcessor.from_pretrained('Qwen/Qwen2-Audio-7B-Instruct')
eval_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-Audio-7B-Instruct', device_map='auto'
)

dataset = load_dataset(
    'PKU-Alignment/EvalAnything-InstructionFollowing',
    name='audio_instruct',
    split='test',
    trust_remote_code=True,
)

with open('.cache/audio_instruct/generated_results.json') as f:
    generated_results = json.load(f)


def parse_response(response: str):
    response = response.upper().strip()

    common_patterns = [
        r'\b(A|B|C|D)\b',
        r'ANSWER[:\s]*(A|B|C|D)\b',
        r'OPTION[:\s]*(A|B|C|D)\b',
        r'CHOICE[:\s]*(A|B|C|D)\b',
    ]

    for pattern in common_patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)

    return ''


def eval_audio_instruct(
    prompt_id: str, question_id: str, question: str, choices: str, answer: str, audio_path: str
):

    conversation = [
        {
            'role': 'system',
            'content': 'You are a helpful assistant, and you need to correctly answer the multiple-choice questions based on the given audio. When answering, you only need to provide the correct option, not the correct content of the options.',
        },
        {
            'role': 'user',
            'content': [
                {'type': 'audio', 'audio_url': audio_path},
                {
                    'type': 'text',
                    'text': USER_PROMPT_TEMPLATE.format(question=question, choice=choices),
                },
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    audio_signal = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)[0]

    inputs = processor(
        text=text,
        audios=audio_signal,
        return_tensors='pt',
        padding=True,
        sampling_rate=processor.feature_extractor.sampling_rate,
    )

    generate_ids = eval_model.generate(**inputs, max_length=256, temperature=0.01)
    generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return {
        'prompt_id': prompt_id,
        'question_id': question_id,
        'score': (
            10 if parse_response(response) == answer else 0
        ),  # make the final score range from 0 to 10
        'response': response,
    }


results = []

for data in tqdm(dataset):
    if data['prompt_id'] not in generated_results:
        continue

    audio_path = generated_results[data['prompt_id']]

    result = eval_audio_instruct(
        prompt_id=data['prompt_id'],
        question_id=data['question_id'],
        question=data['question'],
        choices=data['choice'],
        answer=data['answer'],
        audio_path=audio_path,
    )

    results.append(result)

score = np.mean([result['score'] for result in results])

with open('.cache/audio_instruct/eval_results.json', 'w') as f:
    json.dump({'score': score, 'results': results}, f, indent=4)
