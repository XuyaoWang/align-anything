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
"""Command line interface for interacting with a multi-modal model."""

import os
import librosa
import argparse
import gradio as gr
import numpy as np
import av
import torch
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.configs.template import ChatTemplate
from align_anything.serve.chatbot import SpecialCommand
from PIL import Image
from io import BytesIO
from align_anything.utils.process_video import read_video_pyav

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_EXAMPLES = [
    {'files': [os.path.join(CURRENT_DIR, 'examples/PKU.jpg')], 'text': 'What is great about this image?'},
    {'files': [os.path.join(CURRENT_DIR, 'examples/boya.jpg')], 'text': 'What are the things I should pay attention to when I visit here?'},
    {'files': [os.path.join(CURRENT_DIR, 'examples/logo.jpg')], 'text': 'What is the university of this logo?'},
]

AUDIO_EXAMPLES = [
    {'files': [os.path.join(CURRENT_DIR, 'examples/drum.wav')], 'text': 'What is the emotion of this drumbeat like?'},
    {'files': [os.path.join(CURRENT_DIR, 'examples/laugh.wav')], 'text': 'Is this laughter evil, and why?'},
    {'files': [os.path.join(CURRENT_DIR, 'examples/scream.wav')], 'text': 'What is the main event of this scream?'},
]

VIDEO_EXAMPLES = [
    {'files': [os.path.join(CURRENT_DIR, 'examples/baby.mp4')], 'text': 'What is the video about?'},
]

def multi_modal_conversation(question: str, modality: str):
    if modality == 'image':
        return [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            },
        ]
    elif modality == 'audio':
        return [
            {
                "role": "user",
                "content": [{"type": "audio", "audio_url": "placeholder"}, {"type": "text", "text": question}],
            },
        ]
    elif modality == 'video':
        return [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": question}],
            },
        ]
    raise NotImplementedError(f"Modality {modality} is not supported")

def question_answering(message: dict, history: list):
    question = message['text']
    multi_modal_info = message['files']
    if question == SpecialCommand.RESET:
        history = []
        return '', history
    # Process the image and question
    conversation = multi_modal_conversation(question, modality)
    history.append(conversation)
    
    formatted_conversation = chat_template.format_chat_sample(conversation)[0]

    if modality == 'image':
        images = []
        for file in multi_modal_info:
            if isinstance(file, str):
                images.append(Image.open(file))
            else:
                images.append(Image.open(file['path']))
        inputs = processor(images=images, text=formatted_conversation, return_tensors="pt", padding=True)
    elif modality == 'audio':
        audios = []
        for file in multi_modal_info:
            if isinstance(file, str):
                audios.append(librosa.load(BytesIO(file).read(), sr=processor.feature_extractor.sampling_rate)[0])
            else:
                audios.append(librosa.load(file['path'], sr=processor.feature_extractor.sampling_rate)[0])
        inputs = processor(audios=audios, text=formatted_conversation, return_tensors="pt", padding=True, sampling_rate=processor.feature_extractor.sampling_rate)
    elif modality == 'video':
        videos = []
        for file in multi_modal_info:
            if isinstance(file, str):
                container = av.open(file)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / 8).astype(int)
                clip = read_video_pyav(container, indices)
                videos.append(clip)
            else:
                container = av.open(file['path'])
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / 8).astype(int)
                clip = read_video_pyav(container, indices)
                videos.append(clip)
        inputs = processor(videos=videos, text=formatted_conversation, return_tensors="pt", padding=True)
    else:
        raise NotImplementedError(f"Modality {modality} is not supported")

    # Perform the inference
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.2)
    
    # Extract the predicted answer
    answer = processor.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    history.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return answer


if __name__ == "__main__":
    # Define the Gradio interface
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--modality", type=str, required=True, choices=['image', 'text', 'audio', 'video'])
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    global processor, model, tokenizer, chat_template, modality
    model, tokenizer, processor = load_pretrained_models(model_name_or_path, auto_device_mapping=True, dtype=torch.float16)

    modality = args.modality
    custom_formatter = (
                model.apply_chat_template
                if hasattr(model, 'apply_chat_template')
                else None
            )
    chat_template = ChatTemplate(formatter=processor, custom_formatter=custom_formatter)

    if modality == 'image':
        examples = IMAGE_EXAMPLES
    elif modality == 'audio':
        examples = AUDIO_EXAMPLES
    elif modality == 'video':
        examples = VIDEO_EXAMPLES
    else:
        examples = []

    iface = gr.ChatInterface(
        fn=question_answering,
        type="messages",
        multimodal=True,
        title="Align Anything Multi-Modal CLI",
        description="Upload multiple modalities info and ask a question related. The AI will try to answer it.",
        examples=examples,
        theme=gr.themes.Ocean(text_size="lg", spacing_size="lg", radius_size="lg",),
    )

    iface.launch(share=True)