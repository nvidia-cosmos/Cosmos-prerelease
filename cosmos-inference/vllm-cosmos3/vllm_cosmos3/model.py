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

"""Subclass of Qwen3VLForConditionalGeneration that strips a configurable
prefix from checkpoint weight names before loading.

The prefix is read from the env var COSMOS3_CKPT_PREFIX (e.g. "my_wrapper.").
"""

import os
from collections.abc import Iterable

import torch
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

_COSMOS3_DROP_TOKENS: tuple[str, ...] = (
    "_moe_gen",
    "llm2vae",
    "vae2llm",
    "time_embedder",
)


def _is_und_tower_weight(name: str) -> bool:
    """Return True if `name` belongs to the understanding (text) tower."""
    return not any(token in name for token in _COSMOS3_DROP_TOKENS)


def _to_qwen3vl_hf_name(name: str) -> str:
    """Wrap flat Qwen3 keys into the nested HF Qwen3-VL namespace.

    Cosmos3's understanding tower is saved with the standalone Qwen3 layout
    (``model.layers.*``, ``model.embed_tokens.*``, ``model.norm.*``,
    ``lm_head.*``), but vLLM's ``Qwen3VLForConditionalGeneration`` expects HF
    Qwen3-VL naming (``model.language_model.layers.*`` etc.). Its
    ``hf_to_vllm_mapper`` then rewrites ``model.language_model.`` to
    ``language_model.model.`` for the runtime module tree.
    """
    if name.startswith("model.") and not name.startswith("model.language_model."):
        return "model.language_model." + name[len("model.") :]
    return name


class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        overrides = getattr(vllm_config.model_config.hf_config, "allow_patterns_overrides", None)
        if overrides:
            self.allow_patterns_overrides = list(overrides)
            if any(p.endswith(".safetensors") for p in self.allow_patterns_overrides):
                vllm_config.load_config.load_format = "safetensors"

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        prefix = os.environ.get("COSMOS3_CKPT_PREFIX", "")
        if prefix:
            prefix = prefix.rstrip(".") + "."

        def _iter() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                stripped = name.removeprefix(prefix) if prefix else name
                if _is_und_tower_weight(stripped):
                    yield _to_qwen3vl_hf_name(stripped), tensor

        loaded = super().load_weights(_iter())

        # The cosmos3 checkpoint has no `visual.*` weights — they don't exist
        # in the OmniMoT understanding tower. Treat them as intentionally
        # uninitialized so vLLM 0.19+ doesn't hard-error on missing weights.
        # This means the ViT runs with random init; safe iff prompts are
        # text-only (the vision pathway is then never invoked).
        for name, _ in self.named_parameters():
            if name.startswith("visual."):
                loaded.add(name)
        return loaded
