#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Internal smoke for mixed_modality_sft_nano (T2V/I2V/V2V SFT on Qwen3-VL-8B,
# 4-GPU FSDP, 20 iters) inside the released cosmos_training tree.
#
# Counterpart of test_action_fdm_sft_nano.sh but:
#   - experiment=mixed_modality_sft_nano
#   - dataset is the Bridge SFT jsonl snapshot under workdir/, so
#     ${DATASET_PATH}/train/video_dataset_file.jsonl resolves
#
# Usage:
#   bash test_internal/test_mixed_modality_sft_nano.sh

set -uo pipefail

WORKDIR="/home/pzeren/lustre/Cosmos-prerelease/cosmos_training"
# get_sft_dataset reads ${DATASET_PATH}/train/video_dataset_file.jsonl.
DATASET_PATH="${DATASET_PATH:-/home/pzeren/lustre/Cosmos-prerelease/workdir/cosmos_opensource/sft_dataset_bridge}"
WAN_VAE_PATH="/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"

# Pretrained checkpoint to warm-start from. Override via:
#   CHECKPOINT_PATH=/your/path bash test_internal/test_mixed_modality_sft_nano.sh
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/maoshengl/inter_ckpts/internal}"

OUTPUT_ROOT="/home/pzeren/lustre/Cosmos-prerelease/training_output"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/mixed_modality_sft_nano_debug.log"

mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -d "$WORKDIR" ]]                                          || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -d "$DATASET_PATH" ]]                                     || { echo "ERROR: DATASET_PATH not found: $DATASET_PATH" >&2; exit 1; }
[[ -f "$DATASET_PATH/train/video_dataset_file.jsonl" ]]      || { echo "ERROR: missing $DATASET_PATH/train/video_dataset_file.jsonl" >&2; exit 1; }
[[ -d "$CHECKPOINT_PATH" ]]                                  || { echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2; exit 1; }

cd "$WORKDIR"
echo ">>> $(date '+%H:%M:%S') WORKDIR:    $WORKDIR"
echo ">>> $(date '+%H:%M:%S') dataset:    $DATASET_PATH"
echo ">>> $(date '+%H:%M:%S') checkpoint: $CHECKPOINT_PATH"
echo ">>> $(date '+%H:%M:%S') log:        $LOG_FILE"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HUB_DISABLE_XET=1

# Determinism: PYTHONHASHSEED must be set before the interpreter starts.
export PYTHONHASHSEED=42

DATASET_PATH="$DATASET_PATH" \
LOGURU_LEVEL=DEBUG IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" PYTHONPATH=. \
    torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
    --config=configs/base/config.py \
    --deterministic \
    -- \
    experiment=mixed_modality_sft_nano \
    "checkpoint.load_path=$CHECKPOINT_PATH" \
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    model.config.tokenizer.bucket_name="" \
    model.config.tokenizer.object_store_credential_path_pretrained="" \
    model.config.vlm_config.tokenizer.config_variant=hf \
    model.config.parallelism.data_parallel_shard_degree=-1 \
    model.config.parallelism.use_torch_compile=false \
    trainer.max_iter=20 \
    checkpoint=local \
    checkpoint.save_iter=10 \
    job.group=debug \
    job.name=mixed_modality_sft_nano_debug \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
