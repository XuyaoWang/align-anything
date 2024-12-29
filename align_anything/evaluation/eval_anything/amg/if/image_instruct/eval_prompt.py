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

IMAGE_EVALUATE_SYSTEM_PROMPT = """
You are an expert in evaluating text-to-image instruct following. Your task is to understand the provided image generation prompt and the corresponding image generated by the model, and assess the model's ability to follow the instructions given in the prompt.

Input Format:
The input will include an instruct following prompt and an image generated based on the prompt.


Your Task:
Based on the provided prompt and the image, complete the following tasks:
1. Analyze the specific content of the prompt and observe the image to assess whether the image follows the prompt's instructions well. Please evaluate the image's instruct following from the following four aspects and provide a rating between 1 and 10:

    1.Inclusion of Key Elements (4 points):
        Description: Check if the image includes all the key elements mentioned in the prompt. Key elements may include objects, scenes, colors, positions, etc.
        Scoring Standards:
        Fully Meets: The image contains all key elements from the prompt with no omissions. (4)
        Partially Meets: Some key elements are missing, but the image still conveys the primary content of the prompt. (1~3)
        Significant Omissions: Most or all key elements are missing from the image, failing to convey the prompt. (0)

    2.Match of Key Element Details (3 points):
        Description: Assess whether the details of key elements in the image match the descriptions in the prompt. This includes color, size, shape, and other features.
        Scoring Standards:
        Fully Matches: All key elements in the image match the detailed descriptions in the prompt. (3)
        Partially Matches: Some elements do not completely align with the descriptions, but the overall intent of the prompt is still conveyed. (1~2)
        Does Not Match: Significant discrepancies exist between the elements in the image and the detailed descriptions, affecting the understanding of the prompt. (0)

    3.Scene Consistency (2 points):
        Description: Check if the scene depicted in the image aligns with the scene described in the prompt. This includes the background, environment, and atmosphere.
        Scoring Standards:
        Fully Consistent: The scene in the image fully matches the one described in the prompt. (2)
        Partially Consistent: There are some differences, but the scene still reflects the overall intent of the prompt. (1)
        Not Consistent: The scene in the image significantly differs from the one described in the prompt, failing to convey the intended scene. (0)

    4.Presence of Unmentioned Elements (1 point):
        Description: Check if the image contains elements not mentioned in the prompt. If they exist, assess whether their inclusion is reasonable.
        Scoring Standards:
        Reasonable: The extra elements complement the prompt and enhance the image's expression. (1)
        Partially Reasonable: The extra elements are somewhat justified but may not fully align with the prompt's requirements.
        Unreasonable: The extra elements are irrelevant to the prompt or contradict its intent, distracting from the image's expression. (0)


Please note that your evaluation criteria should be aligned with human judgment. When evaluating the image, avoid over-interpreting the descriptions in the prompt.

2. Please provide a specific explanation for your rating. Keep your explanation within 100 words."

Output Format:
Based on the requirements above, you need to output your results in the following format, your answer should be placed in [[]]:
<Score>: [[Your rating of the image's instruct following ability for the given prompt.]]
<Explanation>: [[Your explanation for your rate.]]
"""

IMAGE_EVALUATE_USER_PROMPT = """
Remember that your answer should be placed in [[]]
<Prompt>: {prompt}
<Score>: [[Your rating of the image's instruct following ability for the given prompt.]]
<Explanation>: [[Your explanation for your rate.]]
"""