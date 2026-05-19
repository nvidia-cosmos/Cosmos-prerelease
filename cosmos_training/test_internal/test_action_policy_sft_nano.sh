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

# Internal smoke for action_policy_sft_nano (LIBERO policy SFT, 4-GPU FSDP,
# 20 iters) inside the released cosmos_training tree.
#
# Counterpart of imaginaire4/test_internal/test_action_policy_sft_nano.sh but:
#   - WORKDIR is the released tree
#   - --config is configs/base/config.py
#   - DATASET_PATH points at the local libero_datasets root, whose subdirs
#     (libero_10 / libero_object / libero_spatial / libero_goal) are looked up
#     by the experiment's ${oc.env:DATASET_PATH}/libero_* paths.
#
# Usage:
#   bash test_internal/test_action_policy_sft_nano.sh

set -uo pipefail

WORKDIR="/home/pzeren/lustre/Cosmos-prerelease/cosmos_training"
DATASET_PATH="${DATASET_PATH:-/nfs/sw/sw_aidot/users/pzeren/imaginaire4/libero_datasets}"
WAN_VAE_PATH="/lustre/fsw/portfolios/cosmos/users/yangyangt/cosmos_opensource/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"

# Pretrained checkpoint to warm-start from. Override via:
#   CHECKPOINT_PATH=/your/path bash test_internal/test_action_policy_sft_nano.sh
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/maoshengl/inter_ckpts/internal}"

OUTPUT_ROOT="/home/pzeren/lustre/Cosmos-prerelease/training_output"
LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/action_policy_sft_nano_debug.log"

mkdir -p "$LOG_DIR"

echo ">>> $(date '+%H:%M:%S') Checking inputs..."
[[ -d "$WORKDIR" ]]         || { echo "ERROR: WORKDIR not found: $WORKDIR" >&2; exit 1; }
[[ -d "$DATASET_PATH" ]]    || { echo "ERROR: DATASET_PATH not found: $DATASET_PATH" >&2; exit 1; }
[[ -d "$CHECKPOINT_PATH" ]] || { echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2; exit 1; }
for suite in libero_10 libero_object libero_spatial libero_goal; do
    [[ -d "$DATASET_PATH/$suite" ]] || { echo "ERROR: LIBERO suite missing: $DATASET_PATH/$suite" >&2; exit 1; }
done

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
    experiment=action_policy_sft_nano \
    "checkpoint.load_path=$CHECKPOINT_PATH" \
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    model.config.tokenizer.bucket_name="" \
    model.config.vlm_config.pretrained_weights.enabled=false \
    model.config.vlm_config.tokenizer.config_variant=hf \
    model.config.parallelism.data_parallel_shard_degree=-1 \
    model.config.parallelism.use_torch_compile=false \
    "++dataloader_train.dataloaders.action_data.dataloader.multiprocessing_context=null" \
    trainer.max_iter=20 \
    trainer.logging_iter=1 \
    checkpoint=local \
    checkpoint.save_iter=10 \
    job.group=debug \
    job.name=action_policy_sft_nano_debug \
    job.wandb_mode=disabled \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo ">>> $(date '+%H:%M:%S') Done (exit $EXIT_CODE)"
exit $EXIT_CODE
