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
# Initialize variables
MODEL_NAME_OR_PATH=""
TRAIN_DATASETS=""
TRAIN_DATA_FILE=""
OUTPUT_DIR=""
JANUS_REPO_PATH=""

export PYTHONPATH=$PYTHONPATH:$JANUS_REPO_PATH
export WANDB_API_KEY=""
export WANDB_MODE=online

# Source the setup script
source ./setup.sh
# Execute deepspeed command
deepspeed \
    --master_port ${MASTER_PORT} \
    --module align_anything.trainers.janus.sft \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_datasets ${TRAIN_DATASETS} \
    --train_data_files ${TRAIN_DATA_FILE} \
    --train_split train \
    --learning_rate 1e-6 \
    --epochs 3 \
    --lr_scheduler_type cosine \
    --output_dir ${OUTPUT_DIR}
