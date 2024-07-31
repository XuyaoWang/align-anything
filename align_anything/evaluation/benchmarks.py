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

import argparse
import json
import os
import re
from typing import Any, Dict, List

import deepspeed
import pandas as pd
import torch
from PIL import Image

from align_anything.evaluation.base import BaseEvaluator
from align_anything.evaluation.categories import GaokaoCategories, MMECategories, MMLUCategories
from align_anything.evaluation.evaluator import GSM8KEvaluator
from align_anything.evaluation.evaluator_registry import get_template_class, register_evaluator
from align_anything.evaluation.mt_bench import MTBench
from align_anything.utils.multi_process import get_current_device, is_main_process
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    read_cfgs,
    seed_everything,
    update_dict,
)
from datasets import Dataset, DatasetDict, load_dataset


MultiPL_E_Categories = {
    'humaneval-clj': 'humaneval-clj',
    'humaneval-cpp': 'humaneval-cpp',
}
@register_evaluator('multipl_e')
class MultiPL_E(BaseEvaluator):
    def get_task_names(self):
        return list(sorted(MultiPL_E_Categories.keys()))

    def get_answer(self, data):
        return 'answer'

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data):
        assert self.num_shot == 0, 'multipl_e does not support few-shot learning.'
        prompt = data['prompt']
        return prompt


@register_evaluator('logiqa')
class LogiQA(BaseEvaluator):
    def get_task_names(self):
        return ['default']

    def get_answer(self, data):
        return 'ABCD'[int(data['correct_option'])]

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data):
        assert self.num_shot == 0, 'logiqa does not support few-shot learning.'
        context = data['context']
        question = data['query']
        choices = data['options']
        prompt = [context+ ' ' + question + ' ' + choice for choice in choices]
        return prompt

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        inputs = [self.processor(prompt, return_tensors='pt').to(self.device) for prompt in prompts]
        answers = self.get_answer(data)
        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }


@register_evaluator('piqa')
class PIQA(BaseEvaluator):
    def get_task_names(self):
        return ['plain_text']

    def get_answer(self, data):
        return 'AB'[int(data['label'])]

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data):
        assert self.num_shot == 0, 'piqa does not support few-shot learning.'
        question = data['goal']
        choices = [data['sol1'],data['sol2']]
        prompt = [question + ' ' + choice for choice in choices]
        return prompt

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        inputs = [self.processor(prompt, return_tensors='pt').to(self.device) for prompt in prompts]
        answers = self.get_answer(data)
        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }

examples = [
    {
        'Question': 'In a 10 Gigabit Ethernet network, the average size of a frame is 1500 bytes. If a burst of noise lasting 1ms interrupts the network, how many frames are lost?',
        'Answer': 'First, calculate the data rate in bytes/s:\n\n10 Gigabit/s * (1 Byte / 8 bits) = 1.25 * 10^9 Bytes/s\n\nNext, calculate the data loss in bytes due to the noise:\n\n1 ms * 1.25 * 10^9 Bytes/s = 1.25 * 10^6 Bytes\n\nFinally, divide the data loss by the average frame size to get the number of frames lost:\n\n1.25 * 10^6 Bytes / 1500 Bytes/frame ≈ 833.33 frames\nThe answer is 833.33'
    },
    {
        'Question': 'Given x = 0.157, what is the value of x \\times \\frac{\\prod_{n=1}^\\infty (1 - \\frac{x^2}{n^2 \\pi^2})}{\\sin(x)}?',
        'Answer': "To evaluate the expression $x \\times \\frac{\\prod_{n=1}^{\\infty} (1 - \\frac{x^2}{n^2 \\pi^2})}{\\sin(x)}$ given x = 0.157, we first recognize that the product in the numerator is related to the sine function through the Euler's reflection formula for the sine function, which can be expressed as:\n\n$$\\sin(x) = x \\prod_{n=1}^{\\infty} \\left(1 - \\frac{x^2}{n^2 \\pi^2}\\right)$$\n\nTherefore, the given expression simplifies to: $x \\times \\frac{\\sin(x)}{\\sin(x)}$\n\nBecause sin(x) in the numerator and denominator cancels out, the expression simplifies further to just x.\n\nSo, given x = 0.157, the value of the expression is 0.157. This result is derived from the properties of the sine function and does not require computational evaluation.\nThe answer is 0.157"
    },
    {
        'Question': 'Consider the basis C of \\mathbb{R}^2 consisting of vectors u_1 = [2, 4] and u_2 = [1, -1]. If y = [8, 12], find the C-coordinate vector of y.',
        'Answer': "The goal is to express y as a linear combination of the basis vectors of C, i.e., $y = a\\cdot u_1 + b\\cdot u_2$, where a and b are the scalar coefficients that we want to find. These coefficients will form the C-coordinate vector of y, which we'll denote as $[a, b]_C$.\n\nGiven:\n- $u_1 = [2, 4]$,\n- $u_2 = [1, -1]$,\n- $y = [8, 12]$.\n\nWe need to solve the system of linear equations:\n2a + 1b = 8\n4a - 1b = 12\n\nLet's solve this system of equations to find a and b.\n\nThe solution to the system of equations is $a = \\frac{10}{3} and b = \\frac{4}{3}$. Therefore, the C-coordinate vector of y in the basis consisting of vectors u_1 = [2, 4] and u_2 = [1, -1] is $\\left[\\frac{10}{3}, \\frac{4}{3}\\right]_C$. \nLet's calculate the numerical value of $\\left[\x0crac{10}{3}, \x0crac{4}{3}\right]_C$ as [3.33, 1.33].\nThe answer is [3.33, 1.33]"
    },
    {
        'Question': 'One can draw a simple, connected planar graph with 200 vertices and 397 edges. Is this statement Trur or False?',
        'Answer': "To determine the answer, we can use Euler's formula for planar graphs, which states that for any finite, connected, planar graph, $V - E + F = 2$, where V is the number of vertices, E is the number of edges, and F is the number of faces.\n\nGiven the modified question, we have V = 200 vertices and E = 397 edges. We want to find if we can have a graph that satisfies these conditions, adhering to Euler's formula.\n\nFirst, let's rearrange Euler's formula to solve for F:  F = E - V + 2\n\nSubstituting the given values: F = 397 - 200 + 2,  F = 199\n\nThis means a graph with 200 vertices and 397 edges would have 199 faces. However, to determine the truth of this possibility, we should check if this graph doesn't violate any other planar graph constraints, particularly regarding the number of edges.\n\nFor a simple, connected planar graph, there's also a relationship between vertices, edges, and faces given by the inequality: $E \\leq 3V - 6$\n\nSubstituting V = 200 gives: $E \\leq 3*200 - 6 = 594$\n\nWith E = 397, the condition $E \\leq 594$ is satisfied, meaning it's theoretically possible in terms of the edge condition for a planar graph.\n\nTherefore, one can draw a simple, connected planar graph with 200 vertices and 397 edges, resulting in 199 faces, without violating the conditions for it to be planar according to both Euler's formula and the constraint on the maximum number of edges.\nThe answer is True"
    },
    {
        'Question': 'Given a finite group G, and a collection of permutations H on a set. Then (a) there always exists H such that G is isomorphic to H; (b) for any H, G is isomorphic to H; (c) G can never be isomorphic to H; (d) none of the above. Which option is correct?',
        'Answer': "This is based on Cayley's theorem, which states that every group G is isomorphic to a subgroup of the symmetric group acting on G. \nIn other words, for every finite group G, there exists a collection of permutations H (which in this context, can be thought of as the set of permutations representing the action of G on itself) such that G is isomorphic to H.\n\nTherefore, there always exists H such that G is isomorphic to H.\nThe answer is (a)"
    }
]
from align_anything.evaluation.utils_theoremqa import answer_clean, compare_answer_with_groundtruth
@register_evaluator('theoremqa')
class TheoremQA(BaseEvaluator):
    def get_task_names(self):
        return ['default']

    def get_answer(self, data):
        if isinstance(data['Answer'], bool):
            answer = [str(data['Answer']), None]
        elif isinstance(data['Answer'], (list, int, float)):
            answer = [str(data['Answer']), data['Answer']]
        else:
            answer = [str(data['Answer']), None]
        return answer

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_example_prompt(self, data, with_answer=True):
        example = "\nProblem:\n" + data['Question'] + '\n' + 'Solution:\n'
        if with_answer:
            example = example + data['Answer'] +'\n'
        return example

    def build_prompt(self, data):
        assert self.num_shot == 0, 'theoremqa does not support few-shot learning.'
        prompts = 'You are supposed to provide a solution to a given problem.\n\n'
        for example in examples:
            prompts = prompts + self.build_example_prompt(example, True)
        # prompts = prompts + self.build_example_prompt(examples[0], True)
        prompts = prompts + self.build_example_prompt(data, False)
        return prompts

    def is_correct(self, pred , answer):
        if pred == None:
            return False
        pred = answer_clean(['The answer is:', 'The answer is', 'the answer is'],pred)
        if isinstance(answer, str):
            answer = [answer]
        return compare_answer_with_groundtruth(pred, *answer)


XCOPACategories = {
    'et': 'Estonian',
    'ht': 'Haitian Creole',
    'id': 'Indonesian',
    'it': 'Italian',
    'qu': 'Southern Quechua',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'th': 'Thai',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'zh': 'Mandarin Chinese',
}
@register_evaluator('xcopa')
class XCOPA(BaseEvaluator):
    def get_task_names(self):
        return list(sorted(XCOPACategories.keys()))

    def get_answer(self, data):
        return 'AB'[int(data['label'])]

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data):
        assert self.num_shot == 0, 'xcopa does not support few-shot learning.'
        question = data['premise']
        choices = [data['choice1'],data['choice2']]
        prompt = [question + ' ' + choice for choice in choices]
        return prompt

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        inputs = [self.processor(prompt, return_tensors='pt').to(self.device) for prompt in prompts]
        answers = self.get_answer(data)
        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }


@register_evaluator('model_Written_evals')
class Model_Written_Evals(BaseEvaluator):
    def get_task_names(self):
        return ['default']

    def get_answer(self, data):
        return data['answer_matching_behavior'][2]

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data):
        assert self.num_shot == 0, 'model_Written_evals does not support few-shot learning.'
        question = data['question']
        choices = ['Answer: A','Answer: B']
        prompt = [question + '\n' + choice for choice in choices]
        return prompt

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        inputs = [self.processor(prompt, return_tensors='pt').to(self.device) for prompt in prompts]
        answers = self.get_answer(data)
        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }


@register_evaluator('toxic_conversations')
class Toxic_Conversation(BaseEvaluator):
    def get_task_names(self):
        return ['default']

    def get_answer(self, data):
        return 'B' if data['label'] else 'A'

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data):
        assert self.num_shot == 0, 'toxic_conversations does not support few-shot learning.'
        question = data['text']
        choices = ['This passage is not toxic','This passage is toxic']
        prompt = [question + '\n' + choice for choice in choices]
        return prompt

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        inputs = [self.processor(prompt, return_tensors='pt').to(self.device) for prompt in prompts]
        answers = self.get_answer(data)
        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }

from typing import Tuple
@register_evaluator('lambada')
class LAMBADA(BaseEvaluator):
    def get_task_names(self):
        return ['plain_text']

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def is_correct(self, pred: str, answer: str) -> bool:
        return pred==answer

    def build_prompt(self, data):
        assert self.num_shot == 0, 'lambada does not support few-shot learning.'
        prompt = data['text']
        return prompt
    
    def choice_ppl(self, inputs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        inputs = inputs['inputs'][0]

        input_ids = inputs['input_ids']
        target_ids = input_ids.clone()
        
        out = self.model(input_ids, labels=target_ids)
        loss = out.loss.to(torch.float32)
        out = out['logits'].argmax(dim=2)[:,-2]

        pred = self.tokenizer.decode(out)
        info = {'candidate_scores': [loss.item()]}
        
        return [pred], [info]
    
    def preproccess(self, data):
        prompts = self.build_prompt(data)
        lim = prompts.rfind(' ')
        answers = prompts[lim+1:]
        prompts = prompts[:lim]
        
        inputs = self.processor(prompts, return_tensors='pt').to(self.device)
        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }


@register_evaluator('mmlu')
class MMLU(BaseEvaluator):
    def get_task_names(self) -> List[str]:
        return list(sorted(MMLUCategories.keys()))

    def get_answer(self, data):
        return chr(65 + data.get('answer', 'N'))

    def set_fewshot_dataset(self, dataset: DatasetDict) -> None:
        self.few_shot_data = dataset['dev']

    def build_example_prompt(self, data, with_answer=True):
        choices = '\n'.join(
            [f'{label}: {data["choices"][ord(label) - 65]}' for label in self.candidate_labels]
        )
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"{data['question']}\n{choices}\n{answer}"

    def build_prompt(self, data):
        prompt = f'The following are multiple choice questions (with answers).\n\n'
        few_shot_examples = self.few_shot_data[: self.num_shot] if self.num_shot else []
        if len(few_shot_examples) == 0:
            return prompt + self.build_example_prompt(data, False)
        else:
            examples = [
                self.build_example_prompt(
                    {key: value[i] for key, value in few_shot_examples.items()}, True
                )
                for i in range(len(few_shot_examples['question']))
            ]
            examples.append(self.build_example_prompt(data, False))
            return prompt + '\n\n'.join(examples)


@register_evaluator('gaokao')
class GaokaoSingleChoice(BaseEvaluator):
    def get_answer(self, data) -> str:
        return data['answer'] if 'answer' in data else 'NoAnswer'

    def set_fewshot_dataset(self, dataset: DatasetDict) -> None:
        self.few_shot_data = dataset['train']

    def load_dataset(self, task_name):
        return load_dataset(
            'json',
            data_files={
                split: os.path.join(self.task_dir, split, GaokaoCategories[task_name])
                for split in ('train', 'dev')
            },
        )

    def get_task_names(self):
        return list(sorted(GaokaoCategories.keys()))

    def build_example_prompt(self, data, with_answer=True):
        question = data['question']
        choices = '\n'.join([f'{label}: {data[label]}' for label in self.candidate_labels])
        answer = f"答案：{data['answer']}" if with_answer else '答案：'
        return f'{question}\n{choices}\n{answer}'

    def build_prompt(self, data):
        few_shot_examples = self.few_shot_data[: self.num_shot] if self.num_shot else []
        if len(few_shot_examples) == 0:
            return self.build_example_prompt(data, False)
        else:
            examples = [
                self.build_example_prompt(
                    {key: value[i] for key, value in few_shot_examples.items()}, True
                )
                for i in range(len(few_shot_examples['question']))
            ]
            examples.append(self.build_example_prompt(data, False))
            return '\n'.join(examples)


@register_evaluator('gsm8k')
class GSM8K(BaseEvaluator):
    gsm8k_evalutor = GSM8KEvaluator()

    def get_task_names(self):
        return ['main']

    def get_answer(self, data):
        return self.gsm8k_evalutor._decimal_separator.sub('', data['answer'])

    def build_example_prompt(self, data, with_answer=True):
        question = data['question']
        prompt = f"Question: {question} Let's think step by step\nAnswer:\n"
        if with_answer:
            return prompt + self.gsm8k_evalutor._decimal_separator.sub('', data['answer'])
        else:
            return prompt

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = dataset['train']

    def build_prompt(self, data):
        prompt = ''
        few_shot_examples = self.few_shot_data[: self.num_shot] if self.num_shot else []
        examples = [
            self.build_example_prompt(
                {key: value[i] for key, value in few_shot_examples.items()}, True
            )
            for i in range(len(few_shot_examples['question']))
        ]
        examples.append(self.build_example_prompt(data, False))
        return prompt + '\n\n'.join(examples)

    def is_correct(self, prediction, reference):
        return self.gsm8k_evalutor.score(prediction, reference)

    def parser_response(self, response):
        response = response[0]
        return response.lstrip()


@register_evaluator('hellaswag')
class Hellaswag(BaseEvaluator):
    def get_task_names(self):
        return ['default']

    def get_answer(self, data):
        return 'ABCDEFG'[int(data['label'])]

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data):
        assert self.num_shot == 0, 'Hellaswag does not support few-shot learning.'
        question = data['ctx']
        choices = data['endings']
        prompt = [question + ' ' + choice for choice in choices]
        return prompt
    

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        inputs = [self.processor(prompt, return_tensors='pt').to(self.device) for prompt in prompts]
        answers = self.get_answer(data)
        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }


@register_evaluator('winogrande')
class Winogrande(BaseEvaluator):
    def get_task_names(self):
        task_names = [
            'winogrande',
        ]
        return task_names

    def load_dataset(self, task_name):
        filename = os.path.join(self.task_dir, 'dev.jsonl')
        with open(filename, encoding='utf-8') as f:
            data = [json.loads(x) for x in f.readlines()]
        for d in data:
            d['id'] = d['qID']
        dataset = DatasetDict(
            {
                'test': Dataset.from_list(data),
            }
        )
        return dataset

    def get_answer(self, data):
        return 'ABCDEFG'[int(data['answer']) - 1] if 'answer' in data else 'NoAnswer'

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data):
        assert self.num_shot == 0, 'Winogrande does not support few-shot learning.'
        question = data['sentence']
        choices = [data[key] for key in ('option1', 'option2')]
        prompt = [question.replace('_', can) for can in choices]
        return prompt

    def preproccess(self, data):
        prompts = self.build_prompt(data)
        inputs = [self.processor(prompt, return_tensors='pt').to(self.device) for prompt in prompts]
        answers = self.get_answer(data)

        return {
            'inputs': inputs,
            'answers': answers,
            'prompts': prompts,
        }


@register_evaluator('mme')
class MME(BaseEvaluator):
    def get_task_names(self):
        return list(MMECategories.keys())

    def load_dataset(self, task_name: str) -> DatasetDict:
        data = []
        questions_dir = os.path.join(self.task_dir, task_name, 'questions_answers_YN')
        image_dir = os.path.join(self.task_dir, task_name, 'images')
        image_files = os.listdir(image_dir)
        for image_file in image_files:
            prefix = image_file.split('.')[0]
            qa_path = os.path.join(questions_dir, prefix + '.txt')
            image_path = os.path.join(image_dir, image_file)

            with open(qa_path) as f:
                text = f.readlines()
            for qa in text:
                question, answer = qa.strip().split('\t')
                data.append(
                    {
                        'question': question,
                        'answer': answer,
                        'image_path': image_path,
                        'id': prefix,
                    }
                )
        return DatasetDict({'val': Dataset.from_list(data)})

    def set_fewshot_dataset(self, dataset):
        self.few_shot_data = None

    def build_prompt(self, data: Dict[str, Any]) -> str:
        assert self.num_shot == 0, 'MME does not support few-shot learning.'
        return f"USER: <image>\n{data['question']}\nASSISTANT: "

    def parser_response(self, response):
        # TODO: batch processing
        response_clean = re.sub(r'[\s\n\t]+', '', response[0]).lower()

        if re.match(r'^yes$', response_clean):
            return 'yes'
        elif re.match(r'^no$', response_clean):
            return 'no'
        else:
            return 'unknown'

    def preproccess(self, data):
        image_path = data['image_path']
        raw_image = Image.open(image_path)
        prompt = self.build_prompt(data)

        inputs = self.processor(prompt, raw_image, return_tensors='pt').to(self.device)

        return {
            'inputs': inputs,
            'answers': data['answer'].lower(),
            'prompts': prompt,
            'id': data['id'],
        }

    def eval_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        details, correction = [], []
        preds, infos = self.predict(instance)

        for id, prompt, answer, pred, info in zip(
            instance['id'], instance['prompts'], instance['answers'], preds, infos
        ):
            is_correct = self.is_correct(pred, answer)

            detail = {
                'id': id,
                'prompt': prompt,
                'pred': pred,
                'answer': answer,
                'is_correct': is_correct,
                **info,
            }

            details.append(detail)
            correction.append(is_correct)
        return details, correction

    def calculate_results(self) -> None:
        acc = {'average': -1}
        total_correct = 0
        total_length = 0
        for task, correction in self.task2correction.items():
            acc[task] = sum(correction) / len(correction) if len(correction) > 0 else -1
            total_correct += sum(correction)
            total_length += len(correction)
        acc['average'] = total_correct / total_length

        acc_plus = {'average': -1}
        total_correct_plus = 0
        total_length_plus = 0
        for task, details in self.task2details.items():
            image_items = {}
            for detail in details:
                if detail['id'] not in image_items:
                    image_items[detail['id']] = {'is_correct': None, 'results': []}
                image_items[detail['id']]['results'].append(detail['is_correct'])
            for id, items in image_items.items():
                image_items[id]['is_correct'] = all(items['results'])
            acc_plus[task] = sum([item['is_correct'] for item in image_items.values()]) / len(
                image_items
            )
            total_correct_plus += sum([item['is_correct'] for item in image_items.values()])
            total_length_plus += len(image_items)
        acc_plus['average'] = total_correct_plus / total_length_plus

        score = {}
        keys = ['average'] + list(self.task2correction.keys())
        for key in keys:
            score[key] = (acc[key] + acc_plus[key]) * 100

        result = {
            'score': score,
            'accuracy': acc,
            'accuracy_plus': acc_plus,
        }

        with open(os.path.join(self.output_dir, self.results_filename), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)


def main():
    # setup distribution
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch.cuda.set_device(current_device)

    # get custom configs from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()

    # read default configs from the yaml file
    task = unparsed_args[-1]
    dict_configs, ds_configs = read_cfgs(mode='evaluation', task=task)

    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))

    configs = dict_to_namedtuple(dict_configs)

    evaluator = get_template_class(task, configs.default, ds_configs)
    evaluator.eval()


if __name__ == '__main__':
    main()
