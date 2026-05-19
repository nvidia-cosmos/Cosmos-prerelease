# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Mixed-modality SFT (T2V / I2V / V2V) on Qwen3-VL-8B with EMA enabled — "nano".

Sibling of ``mixed_modality_sft_8b``: same Qwen3-VL-8B backbone and 70/20/10
T2V/I2V/V2V conditioning mix, but turns EMA on for the generation pathway.

Both ``checkpoint.load_path`` and the dataset ``jsonl_paths`` are required
overrides (upstream YAML had them as Hydra ``???`` placeholders); supply them
on the CLI or in a downstream experiment that inherits from this one.

Usage::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 \\
        --master_port=12341 -m scripts.train \\
        --config=configs/base/config.py -- \\
        experiment=mixed_modality_sft_nano \\
        checkpoint.load_path=<path> \\
        'dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=[<path>]'
"""

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict

from configs.base.defaults.vlm import (
    create_qwen2_tokenizer_with_download,
    create_vlm_config,
)
from cosmos.data.vfm.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos.data.vfm.local_datasets.sft_dataset import get_sft_dataset
from cosmos.model.vfm.mot.unified_mot import Qwen3VLTextConfig, Qwen3VLTextForCausalLM

cs = ConfigStore.instance()


mixed_modality_sft_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                    "generation",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /cluster": "gcp_iad_gb200"},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="sft",
            name="mixed_modality_sft_nano",
            wandb_mode="disabled",
        ),
        data_setting=dict(
            qwen_max_video_token_length=8192,
        ),
        model=dict(
            config=dict(
                action_gen=True,
                causal_training_strategy="none",
                input_caption_key="ai_caption",
                input_image_key="images",
                input_video_key="video",
                joint_attn_implementation="two_way",
                latent_downsample_factor=16,
                log_enc_time_every_n=100,
                max_action_dim=64,
                max_num_tokens_after_packing=45056,
                num_embodiment_domains=32,
                resolution="720",
                sound_gen=False,
                sound_latent_fps=25,
                state_ch=48,
                state_t=300,
                video_temporal_causal=False,
                vision_gen=True,
                diffusion_expert_config=dict(
                    base_fps=24,
                    enable_fps_modulation=True,
                    load_weights_from_pretrained=True,
                    max_vae_latent_side_after_patchify=20,
                    patch_spatial=2,
                    position_embedding_type="unified_3d_mrope",
                    rope_h_extrapolation_ratio=1.0,
                    rope_t_extrapolation_ratio=1.0,
                    rope_w_extrapolation_ratio=1.0,
                    timestep_range=1.0,
                    unified_3d_mrope_reset_spatial_ids=True,
                    unified_3d_mrope_temporal_modality_margin=15000,
                ),
                ema=dict(
                    enabled=True,
                    iteration_shift=0,
                    rate=0.1,
                ),
                lbl=dict(
                    coeff_gen=None,
                    coeff_und=None,
                    method="local",
                ),
                parallelism=dict(
                    cfg_parallel_shard_degree=1,
                    compile_dynamic=True,
                    compiled_region="language",
                    context_parallel_shard_degree=1,
                    coordinate_descent_tuning=False,
                    data_parallel_shard_degree=8,
                    enable_inference_mode=False,
                    max_autotune_pointwise=False,
                    precision="bfloat16",
                    use_activation_checkpointing=True,
                    use_cuda_graphs=False,
                    use_torch_compile=True,
                ),
                rectified_flow_inference_config=dict(
                    num_train_timesteps=1000,
                    scheduler_type="unipc",
                    shift=1,
                    use_dynamic_shifting=False,
                ),
                rectified_flow_training_config=dict(
                    action_loss_weight=10.0,
                    high_sigma_ratio=0.05,
                    high_sigma_timesteps_max=1000,
                    high_sigma_timesteps_min=995,
                    image_loss_scale=1.0,
                    independent_action_schedule=False,
                    loss_scale=1.0,
                    normalize_loss_by_active=False,
                    shift={"256": 3, "480": 5, "720": 10},
                    train_time_action_distribution="logitnormal",
                    train_time_image_distribution="logitnormal",
                    train_time_sound_distribution="logitnormal",
                    train_time_video_distribution="waver",
                    train_time_weight="uniform",
                    use_discrete_rf=False,
                    use_dynamic_shift=False,
                    use_high_sigma_strategy=False,
                    use_high_sigma_strategy_action=False,
                ),
                tokenizer=dict(
                    bucket_name="bucket",
                    chunk_duration=93,
                    encode_chunk_frames={"256": 68, "480": 24, "720": 12},
                    encode_exact_durations=None,
                    keep_decoder_cache=False,
                    object_store_credential_path_pretrained="credentials/gcp_checkpoint.secret",
                    spatial_compression_factor=16,
                    temporal_compression_factor=4,
                    use_streaming_encode=False,
                    vae_path="pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
                ),
                vlm_config=dict(
                    layer_module="Qwen2MoTDecoderLayer",
                    model_name="Qwen/Qwen3-VL-8B-Instruct",
                    tie_word_embeddings=False,
                    use_system_prompt=False,
                    pretrained_weights=dict(
                        enabled=False,
                        backbone_path="s3://bucket/cosmos3/pretrained/huggingface/Qwen/Qwen3-VL-8B-Instruct/",
                        credentials_path="credentials/gcp_checkpoint.secret",
                        enable_gcs_patch_in_boto3=True,
                    ),
                    model_instance=L(Qwen3VLTextForCausalLM)(
                        config=L(create_vlm_config)(
                            base_config=L(Qwen3VLTextConfig.from_json_file)(
                                json_file=(
                                    "cosmos/model/vfm/vlm/qwen3_vl/configs/"
                                    "Qwen3-VL-8B-Instruct.json"
                                ),
                            ),
                            freeze_und=False,
                            layer_module="MoTDecoderLayer",
                            qk_norm_for_text=True,
                            tie_word_embeddings=True,
                        ),
                    ),
                    tokenizer=L(create_qwen2_tokenizer_with_download)(
                        config_variant="gcp",
                        pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
                    ),
                ),
            ),
        ),
        optimizer=dict(
            betas=[0.9, 0.95],
            eps=1.0e-06,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
            ],
            lr=2.0e-05,
            lr_multipliers={},
            weight_decay=0,
        ),
        scheduler=dict(
            cycle_lengths=[1000],
            f_max=[1.0],
            f_min=[0.0],
            f_start=[0.0],
            verbosity_interval=0,
            warm_up_steps=[50],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=2,
            logging_iter=1,
            max_iter=500,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                compile_tokenizer=dict(
                    compile_after_iterations=3,
                    enabled=False,
                    warmup_resolutions=None,
                ),
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200,
                    log_memory_detail=True,
                    save_s3=False,
                    step_size=1,
                    upload_every_n_mul=5,
                ),
                every_n_sample_ema=dict(
                    do_x0_prediction=False,
                    every_n=999999,
                    fps=16,
                    guidance=[0.0, 3.0, 7.0],
                    is_ema=True,
                    n_sample_to_save=128,
                    n_sigmas_for_x0_prediction=4,
                    n_viz_sample=2,
                    num_sampling_step=35,
                    prompt_type="t5_xxl",
                    run_at_start=False,
                    save_local=False,
                    save_s3=False,
                    step_size=1,
                    use_negative_prompt=False,
                ),
                every_n_sample_reg=dict(
                    do_x0_prediction=False,
                    every_n=999999,
                    fps=16,
                    guidance=[0.0, 3.0, 7.0],
                    is_ema=False,
                    n_sample_to_save=128,
                    n_sigmas_for_x0_prediction=4,
                    n_viz_sample=2,
                    num_sampling_step=35,
                    prompt_type="t5_xxl",
                    run_at_start=False,
                    save_local=False,
                    save_s3=False,
                    step_size=1,
                    use_negative_prompt=False,
                ),
                expert_heatmap=dict(every_n=1000),
                grad_clip=dict(clip_norm=0.1, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                norm_monitor=dict(
                    every_n=100,
                    layer_norm_only=False,
                    log_stat_wandb=True,
                    save_s3=False,
                    step_size=1,
                    track_activations=True,
                ),
                param_count=dict(save_s3=False),
                sequence_packing_padding=dict(every_n=1),
                sigma_loss_analysis=dict(every_n=500, every_n_viz=500, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                wandb_2x=dict(
                    logging_iter_multipler=1,
                    save_logging_iter_multipler=1,
                    save_s3=False,
                ),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_path="???",  # supply via CLI / downstream experiment
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=True,
            verbose=True,
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="default",
            max_samples_per_batch=None,
            max_sequence_length=45056,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=True,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                datasets=dict(
                    video=dict(
                        ratio=1,
                        dataset=L(get_sft_dataset)(
                            append_duration_fps_timestamps=True,
                            append_resolution_info=True,
                            caption_suffix="",
                            cfg_dropout_keep_metadata=False,
                            cfg_dropout_rate=0.1,
                            # 70% T2V, 20% I2V (first frame), 10% V2V (first 5 frames / 2 latent frames)
                            conditioning_config={0: 0.7, 1: 0.2, 2: 0.1},
                            conditioning_fps=-1,
                            conditioning_fps_noise_std=0.0,
                            frame_selection_mode="first",
                            jsonl_paths=["${oc.env:DATASET_PATH}/train/video_dataset_file.jsonl"],
                            min_short_edge=0,
                            num_video_frames=-1,
                            resolution="256",
                            sample_by_window=False,
                            temporal_compression_factor=4,
                            temporal_interval_mode="max_30fps",
                            use_system_prompt=False,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [mixed_modality_sft_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    _item["job"]["name"] = _name + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
