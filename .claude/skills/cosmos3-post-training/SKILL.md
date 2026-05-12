---
name: cosmos3-post-training
description: >
  Guide users through Cosmos3 supervised fine-tuning (SFT) post-training:
  preparing the example dataset and DCP base checkpoint, editing the experiment
  config, launching distributed training with `torchrun`, running T2V/I2V/V2V
  inference with the trained DCP checkpoint, optionally exporting it to
  Hugging Face safetensors, and the optional Video Captioning pipeline for
  building custom datasets. Use when the user asks "how do I post-train Cosmos3", "how do I
  fine-tune on my own video dataset", "how do I export a trained checkpoint",
  "how do I caption videos for training", or any question about SFT, the
  `cu130-train` / `cu128-train` install groups, the `mixed_modality_sft_8b.yaml` config,
  the `convert_model_to_dcp` / `export_model` / `train` / `caption_from_video`
  / `captions_to_sft_jsonl` scripts, or where SFT outputs land on disk.
---

# Cosmos3 Post-Training (SFT)

## When to use this skill

- User wants to fine-tune Cosmos3-Nano on their own video dataset (SFT)
- User asks which fields in `mixed_modality_sft_8b.yaml` to override (lr, FSDP shard, max_iter, jsonl_paths, ...)
- User wants to convert a base Hugging Face checkpoint to DCP, or convert a trained DCP back to safetensors
- User wants to caption raw videos with a VLM to build a training dataset
- User wants to assemble a JSONL manifest from videos + captions
- For installation, `--group=cu130-train` / `cu128-train`, or LD_LIBRARY_PATH issues, hand off to **cosmos3-setup**
- For inference parameters, parallelism presets, or online serving, hand off to **cosmos3-inference**

## Path convention

All paths below are relative to the cosmos3 package root (`../../../` from this skill file). All `uv run` / `python` / `torchrun` commands should also be run from there.

## Where to find answers

The canonical reference is `docs/training.md`. Use this table to route questions:

| User question                                                      | Go to                                                               |
| ------------------------------------------------------------------ | ------------------------------------------------------------------- |
| Full step-by-step SFT workflow                                     | `docs/training.md`                                                  |
| Which install group? (`cu130-train` vs `cu128-train`)              | `docs/training.md` § Setup, `docs/setup.md` § CUDA Variants         |
| How do I download the example bridge dataset?                      | `docs/training.md` § Step 1 — Prepare data and checkpoint           |
| How do I convert a base HF checkpoint to DCP?                      | `docs/training.md` § Step 1 — Prepare data and checkpoint           |
| Which `mixed_modality_sft_8b.yaml` fields are commonly overridden? | `docs/training.md` § Step 2 — Prepare config                        |
| How do I launch distributed training?                              | `docs/training.md` § Step 3 — Run training                          |
| How do I validate the config without actually training?            | `docs/training.md` § Step 3 (the `--dry-run` flag)                  |
| How do I export the trained DCP back to safetensors?               | `docs/training.md` § Export checkpoint to Hugging Face safetensors  |
| How do I run inference with the trained checkpoint?                | `docs/training.md` § Run inference with trained checkpoint          |
| Where do training artifacts land?                                  | `docs/training.md` § Outputs                                        |
| How do I caption raw videos for SFT?                               | `docs/training.md` § Video Captioning for Training Data Processing  |
| How do I serve the captioning VLM?                                 | `docs/training.md` § Server setup                                   |
| How do I build a JSONL dataset from captions + videos?             | `docs/training.md` § Creating Video Dataset JSONL File for Training |

## Workflow at a glance

1. **Setup** — install the training extras: `uv sync --all-extras --group=cu130-train` (or `cu128-train` on older drivers), then `source .venv/bin/activate && export LD_LIBRARY_PATH=`.
2. **Step 1 — Prepare data and checkpoint** — download the example bridge dataset to `$DATASET_PATH` (Hugging Face cache) and `convert_model_to_dcp` the base checkpoint into `$BASE_CHECKPOINT_PATH` (default: `/tmp/$USER/checkpoints/cosmos3_nano`).
3. **Step 2 — Prepare config** — the provided `cosmos3/configs/experiment/mixed_modality_sft_8b.yaml` runs as-is on the example dataset (~100 iterations); override `model.config.parallelism.data_parallel_shard_degree`, `dataloader_train.dataloader.datasets.*.jsonl_paths`, `optimizer.lr`, `trainer.max_iter`, etc. for custom runs.
4. **Step 3 — Run training** — `torchrun --nproc_per_node=8 -m cosmos3.scripts.train -o outputs/train --config-file cosmos3/configs/experiment/mixed_modality_sft_8b.yaml --config-overrides "checkpoint.load_path=$BASE_CHECKPOINT_PATH" "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=$DATASET_PATH/train/video_dataset_file.jsonl"` (use `--dry-run` first when iterating on config). DCP checkpoints land in `outputs/train/job/checkpoints/iter_<N>`.
5. **Inference** — read `outputs/train/job/checkpoints/latest_checkpoint.txt`, point `cosmos3.scripts.inference` at the resulting `outputs/train/job/checkpoints/iter_<N>` DCP path with `--config-file outputs/train/config.yaml`. The example input glob `"$DATASET_PATH/val/inference_prompt*/episode_049683_clip000.json"` covers T2V, I2V, and V2V (see `cosmos3-inference` skill for presets / input formats).
6. **Export (optional)** — `cosmos3.scripts.export_model` converts the DCP iter to Hugging Face safetensors at `outputs/train/model`. Not required for the standard inference flow above.

## Things not obvious from the docs

- **Training extras are a separate group**: SFT requires the `cu130-train` / `cu128-train` install group, not the inference-only `cu130` / `cu128`. Re-running `uv sync` with the wrong group silently leaves training deps uninstalled.
- **`-o` controls the entire output tree**: passing `-o outputs/train` to `cosmos3.scripts.train` makes everything land under `outputs/train/job/...` (logs, `config.yaml`, `checkpoints/iter_<N>`, callback outputs). Without `-o`, training falls back to `${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/{job.project}/{job.group}/{job.name}/`.
- **Inference uses the DCP checkpoint directly**: the standard flow points `cosmos3.scripts.inference` at `outputs/train/job/checkpoints/iter_<N>` together with `--config-file outputs/train/config.yaml`. The Hugging Face safetensors export (`outputs/train/model`) is optional — only needed if you want a portable single-file checkpoint.
- **Mixed-modality input glob**: the example uses `"$DATASET_PATH/val/inference_prompt*/episode_049683_clip000.json"` with a `*` so a single command runs T2V, I2V, and V2V (the dataset has `inference_prompt/`, `inference_prompt_i2v/`, `inference_prompt_v2v/` siblings under `val/`).
- **`data_parallel_shard_degree` must equal `WORLD_SIZE`**: it has to match `--nproc_per_node` on the `torchrun` command. Mismatch → FSDP init failure.
- **`--dry-run`**: `cosmos3.scripts.train` accepts `--dry-run` to validate the config end-to-end without launching training. Use it whenever iterating on YAML overrides.
- **Captioning server flags**: `vllm serve ... --allowed-local-media-path /` is required so the VLM can read the `file://` paths the captioning script sends. Use `Qwen/Qwen3-VL-8B-Instruct-FP8` as the recommended model; first launch downloads weights and may take several minutes (server is ready when you see `Application startup complete.`).
- **Captioning input modes**: `cosmos3.scripts.caption_from_video` accepts `--video <file_or_dir>` (single file or directory of `.mp4`s) or `-i <jsonl>` where each line has a `vision_path` field — same JSONL format used downstream by training.
- **Captioning output layout**: each input video produces a directory containing `caption.txt` and `sample_args.json`; `captions_to_sft_jsonl` then assembles those plus the source videos into a training-ready JSONL.
- **`uv run` over bare `python`**: per repo convention, prefer `uv run` for new commands. The `python -m cosmos3.scripts.caption_from_video ...` snippets in `docs/training.md` are a known pending migration.

## Related skills

| Skill                                  | When to use                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| `../cosmos3-setup/SKILL.md`            | Initial install, CUDA variant selection, container/`LD_LIBRARY_PATH` setup   |
| `../cosmos3-inference/SKILL.md`        | Inference parameters, parallelism presets, input JSON format, online serving |
| `../cosmos3-codebase-nav/SKILL.md`     | Locating configs, scripts, and defaults inside the package                   |
| `../cosmos3-env-troubleshoot/SKILL.md` | Debugging environment / runtime errors during training                       |
