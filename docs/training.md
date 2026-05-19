# Post-Training (Supervised Fine-Tuning)

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Training](#training)
  - [Step 1 - Prepare data and config](#step-1---prepare-data-and-config)
  - [Step 2 — Prepare checkpoint](#step-2--prepare-checkpoint)
  - [Step 3 — Run training](#step-3--run-training)
    - [VFM Post-Training](#vfm-post-training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Config](#config)

______________________________________________________________________

<!--TOC-->

Fine-tune a pre-trained Cosmos3 model on your own dataset using supervised fine-tuning (SFT).

This guide was tested on the following environments:

- 8x H100 (80 GB)

Prerequisites:

- [Setup](../README.md#setup)
- [Environment Variables](../cosmos-inference/docs/environment_variables.md)

## Training

### Step 1 - Prepare data and config

Datasets are downloaded to the [Hugging Face cache](https://huggingface.co/docs/datasets/cache).

**Note:** Some datasets are license gated; visit the repository page and accept any license terms. Ensure you are authenticated with `uvx hf@latest auth login`.

Select one of the following recipes. Each sets a flat interface `--toml=<file>` (the same pattern as `examples/launch_mixed_modality_sft_8b_toml.sh`). `--config` defaults to `configs/base/config.py` (vfm-base); pass it explicitly only for vfm-vlm (`configs/base/vlm/config.py`).

<details open><summary><b>Vision Generation with JSONL Dataset</b></summary>

Fine-tune vision generation on [nvidia/bridge-v2-subset-synthetic-captions](https://huggingface.co/datasets/nvidia/bridge-v2-subset-synthetic-captions/tree/main).

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano
TOML_FILE="examples/toml/launch_mixed_modality_sft_8b.toml"

export DATASET_PATH=$(uvx hf@latest download --repo-type dataset nvidia/bridge-v2-subset-synthetic-captions --revision 46468e12ac0dd36901e9e3240d4fc7620942b5d7 --quiet)/sft_dataset_bridge/train/video_dataset_file.jsonl
export WAN_VAE_PATH=/path/to/wan2pt2/Wan2.2_VAE.pth

EXTRA_OVERRIDES=(
  "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=[\"$DATASET_PATH\"]"
  "model.config.tokenizer.vae_path=$WAN_VAE_PATH"
)
```

For more details, see [JSONL Dataset](../cosmos-inference/docs/dataset_jsonl.md).

</details>

<details><summary><b>Action Policy with LIBERO LeRobot Dataset</b></summary>

Fine-tune action policy on [nvidia/LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3/). This config trains in policy mode with concatenated `agentview` and wrist camera observations, frame-wise relative actions, 6D rotations, and quantile-rotation action normalization.

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano
TOML_FILE="examples/toml/launch_action_libero.toml"

export LIBERO_LOCAL_DATA_ROOT=$(uvx hf@latest download --repo-type dataset nvidia/LIBERO_LeRobot_v3 --revision ddc1edeb6e51e2b7d4d2ba7a1433daaecd37aa64 --quiet)
export WAN_VAE_PATH=/path/to/wan2pt2/Wan2.2_VAE.pth

EXTRA_OVERRIDES=(
  "model.config.tokenizer.vae_path=$WAN_VAE_PATH"
)
```

For evaluation, see [LIBERO Closed-Loop Evaluation](../cosmos-inference/docs/action_policy_closed_loop_eval.md).

</details>

<details><summary><b>Text-to-World with Local DataPacker</b></summary>

Fine-tune the text-conditioned video generation pathway with the local
DataPacker pipeline.

```shell
BASE_CHECKPOINT_NAME=Cosmos3-Nano
TOML_FILE="examples/toml/launch_t2w_sft_local_datapacker.toml"

export DATASET_PATH=/path/to/your/sft_t2w_dataset.jsonl
export WAN_VAE_PATH=/path/to/wan2pt2/Wan2.2_VAE.pth

EXTRA_OVERRIDES=(
  "dataloader_train.data_source.jsonl_paths=[\"$DATASET_PATH\"]"
  "model.config.tokenizer.vae_path=$WAN_VAE_PATH"
)
```

</details>

### Step 2 — Prepare checkpoint

Convert base checkpoint to [PyTorch Distributed Checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html) format:

```shell
BASE_CHECKPOINT_PATH=/tmp/$USER/checkpoints/$BASE_CHECKPOINT_NAME
python -m cosmos3.scripts.convert_model_to_dcp \
  -o $BASE_CHECKPOINT_PATH \
  --checkpoint-path $BASE_CHECKPOINT_NAME
```

Set the resulting path under `[train.ckpt].load_path` in your `$TOML_FILE`, or pass it as a trailing positional Hydra override (as the example launchers do).

### Step 3 — Run training

#### VFM Post-Training

Launch distributed training with `torchrun` (TOML interface mode, as in `examples/launch_mixed_modality_sft_8b_toml.sh`):

```shell
cd cosmos_training
IMAGINAIRE_OUTPUT_ROOT=outputs/train PYTHONPATH=. \
torchrun --nproc_per_node=8 -m scripts.train \
    --toml=../$TOML_FILE \
    -- \
    "checkpoint.load_path=$BASE_CHECKPOINT_PATH" \
    "${EXTRA_OVERRIDES[@]}"
```

The `--toml` file is translated to Hydra overrides by `cosmos_training/scripts/interface_toml.py`; the variant (`vfm-base` vs `vfm-vlm`) is inferred from the `--config` path. Per-user runtime paths (datasets, VAE, DCP base) flow through the trailing positional args after `--`, which is also where you append any one-off Hydra overrides.

For vfm-vlm experiments, pass `--config=configs/base/vlm/config.py` explicitly (see `examples/launch_vlm_llava_ov_toml.sh`).

Arguments:

- `--config`: Python config entry point. Defaults to `configs/base/config.py` (vfm-base); pass `configs/base/vlm/config.py` for vfm-vlm.
- `--toml`: Flat interface TOML translated to Hydra overrides via `cosmos_training/scripts/interface_toml.py`. See `examples/toml/launch_*.toml` for ready-to-use templates.
- Positional arguments after `--`: [Hydra overrides](https://hydra.cc/docs/advanced/override_grammar/basic/). See [Config](#config).
- `--dryrun`: Validate the setup and emit `config.yaml` without running training.

To resume from the latest in-progress checkpoint, point `[train.ckpt].load_path` (or `checkpoint.load_path` as a trailing override) at the run's `job/checkpoints/iter_<N>/` directory.

Outputs (under `$IMAGINAIRE_OUTPUT_ROOT/<project>/<group>/<name>/`):

1. `debug.log`, `console.log`: Logs.
1. `config.yaml`: Finalized hydra config.
1. `job/`
    1. `checkpoints/`
        1. `latest_checkpoint.txt`: Text file containing the latest checkpoint iteration.
        1. `iter_<iter>/`: DCP checkpoints saved every `[train.ckpt].save_freq` iterations.
    1. `<callback_name>/`: Callback outputs.

## Inference

Export the DCP checkpoint to a Hugging Face safetensors checkpoint:

```shell
CHECKPOINT_ITER=$(cat outputs/train/job/checkpoints/latest_checkpoint.txt)
CHECKPOINT_PATH=outputs/train/job/checkpoints/$CHECKPOINT_ITER

python -m cosmos3.scripts.export_model \
  --checkpoint-path $CHECKPOINT_PATH \
  --config-file outputs/train/config.yaml \
  -o outputs/train/model
```

The checkpoint can be used in [Inference](../README.md#inference) commands by passing `--checkpoint-path outputs/train/model`.

## Evaluation

**Supported modalities:** Forward Dynamics, Inverse Dynamics, Policy.

Run inference on held-out dataset split and compare to ground truth:

```shell
torchrun --nproc-per-node=8 -m cosmos3.scripts.eval \
    -o outputs/train_eval \
    --checkpoint-path outputs/train/model \
    --dataset.config-file outputs/train/config.yaml
```

Arguments:

- `--dataset.model-mode`: Which modality to evaluate.
- `--dataset.num-samples N`: Maximum number of samples to evaluate.

Outputs:

- `metrics_aggregate.json`: Aggregate metrics.
- `<dataset>/<mode>/<id>/metrics.json`: Per sample metrics.

Metrics:

- Vision: Peak Signal-to-Noise Ratio (PSNR)
- Action: Mean Squared Error (MSE)

## Config

The following fields are commonly tuned. The left column is the TOML interface key (the canonical knob in `$TOML_FILE`); the right column is the Hydra dotted path you'd use as a trailing positional override after `--`. The full mapping lives in `cosmos_training/scripts/interface_toml.py`.

1. `[job]`
    1. `project`, `group`, `name` → `job.project` / `job.group` / `job.name` — Job directory path components.
    1. `wandb_mode` → `job.wandb_mode` — Enable Wandb logging (`online`, `disabled`). `online` requires `WANDB_API_KEY` in env.
1. `[policy.parallelism]`
    1. `dp_shard_size` → `model.config.parallelism.data_parallel_shard_degree` — FSDP shard size. `dp_shard_size × dp_replicate_size × cp_size` must equal `WORLD_SIZE`.
    1. `dp_replicate_size` → `model.config.parallelism.data_parallel_replicate_degree` — HSDP replicate degree.
    1. `cp_size` → `model.config.parallelism.context_parallel_shard_degree` — Context-parallel shard size.
1. `[train]`
    1. `compile` → `model.config.parallelism.use_torch_compile` — Enable torch compile. Faster, more memory.
    1. `max_iter` → `trainer.max_iter` — Total number of training iterations.
    1. `param_dtype` → `model.config.parallelism.precision` — `bfloat16` / `float16` / `float32`.
1. `[train.train_policy]`
    1. `experiment` → `experiment` — Registered SKU name (e.g. `mixed_modality_sft_8b`).
    1. `mini_batch` → `trainer.grad_accum_iter` (vfm-base) / `data_setting.max_batch_size` (vfm-vlm) — Gradient accumulation / micro-batch size.
1. `[train.ckpt]`
    1. `load_path` → `checkpoint.load_path` — Base DCP checkpoint path.
    1. `save_freq` → `checkpoint.save_iter` — Save DCP checkpoint every N iterations.
    1. `save_mode` → `checkpoint.dcp_async_mode_enabled` (`async` → true, `sync` → false).
1. `[policy]`
    1. `model_name_or_path` → `model.config.vlm_config.model_name` (vfm-base) / `model.config.policy.backbone.model_name` (vfm-vlm) — HF identifier or local snapshot.
    1. `model_gradient_checkpointing` → `model.config.{parallelism,policy.parallelism}.use_activation_checkpointing`.

Knobs without a TOML interface key today (pass as trailing positional Hydra overrides):

- `dataloader_train.dataloader.datasets.*.dataset.jsonl_paths` — dataset JSONL file paths.
- `dataloader_train.data_source.num_video_frames` — number of input video frames to sample (`-1`: all frames).
- `model.config.tokenizer.vae_path` — Wan2.2 VAE `.pth` snapshot.
- `model.config.vlm_config.pretrained_weights.enabled` — toggle the HF backbone overlay.
- `model.config.vlm_config.tokenizer.config_variant` — `hf` / `s3` / `gcp` for the tokenizer download source.
- `checkpoint.load_from_object_store.enabled` / `checkpoint.save_to_object_store.enabled` — local FS vs. object store.
- `optimizer.lr` — base learning rate (the TOML init_lr / end_lr are folded into the scheduler ratios, not a direct LR scalar).
