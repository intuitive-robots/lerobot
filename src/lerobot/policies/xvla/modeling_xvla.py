#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from __future__ import annotations

import builtins
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import threading

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_STATE

from .action_hub import build_action_space
from .configuration_florence2 import Florence2Config
from .configuration_xvla import XVLAConfig
from .modeling_florence2 import Florence2ForConditionalGeneration
from .soft_transformer import SoftPromptedTransformer


class XVLATimingProfiler:
    """
    Profiler for measuring timing of different XVLA components.
    Stores measurements and saves averages to a text file.

    Components measured:
    - vlm_vision_encoder: Time for image encoding via Florence2 vision tower
    - vlm_text_embedding: Time for text token embedding
    - vlm_multimodal_merge: Time for merging image and text features
    - vlm_language_encoder: Time for language model encoder forward pass
    - forward_vlm_total: Total VLM encoding time
    - action_preprocessing: Time for noise addition and action space preprocessing
    - policy_transformer_forward: Time for policy transformer inference
    - loss_computation: Time for loss calculation
    - forward_total: Total forward pass time
    - denoising_loop_total: Total time for all denoising steps during inference
    - denoising_step_N: Time for each individual denoising step
    - denoising_step_avg: Average time per denoising step
    - action_postprocessing: Time for action postprocessing
    - generate_actions_total: Total action generation time
    - batch_preparation: Time to prepare batch inputs
    - select_action_total: Total time for action selection
    """

    def __init__(self, output_file: str = "xvla_timing_profile.txt", save_interval: int = 100):
        self.output_file = output_file
        self.save_interval = save_interval
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.call_count = 0
        self._lock = threading.RLock()
        self._cuda_available = torch.cuda.is_available()

    def _sync_cuda(self) -> None:
        if self._cuda_available:
            torch.cuda.synchronize()

    def start_timer(self) -> float:
        self._sync_cuda()
        return time.perf_counter()

    def end_timer(self, start_time: float, component_name: str) -> float:
        self._sync_cuda()
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
        should_save = False
        with self._lock:
            self.timings[component_name].append(elapsed)
            self.call_count += 1
            if self.call_count % self.save_interval == 0:
                should_save = True
        # Save outside lock to avoid blocking other threads during I/O
        if should_save:
            self.save_to_file()
        return elapsed

    def record(self, component_name: str, elapsed_ms: float) -> None:
        with self._lock:
            self.timings[component_name].append(elapsed_ms)

    def get_stats(self) -> dict[str, dict[str, float]]:
        stats = {}
        with self._lock:
            # Make a copy of timings to avoid holding lock during computation
            timings_copy = {k: list(v) for k, v in self.timings.items()}
        for name, times in timings_copy.items():
            if times:
                stats[name] = {
                    "count": len(times),
                    "mean_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "total_ms": sum(times),
                }
        return stats

    def save_to_file(self) -> None:
        stats = self.get_stats()
        if not stats:
            return

        with open(self.output_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("XVLA Timing Profile - Component Statistics\n")
            f.write("=" * 70 + "\n\n")

            # Sort by mean time descending
            sorted_stats = sorted(stats.items(), key=lambda x: x[1]["mean_ms"], reverse=True)

            for name, data in sorted_stats:
                f.write(f"Component: {name}\n")
                f.write(f"  Calls:    {data['count']}\n")
                f.write(f"  Mean:     {data['mean_ms']:.4f} ms\n")
                f.write(f"  Min:      {data['min_ms']:.4f} ms\n")
                f.write(f"  Max:      {data['max_ms']:.4f} ms\n")
                f.write(f"  Total:    {data['total_ms']:.4f} ms\n")
                f.write("-" * 40 + "\n")

            # Summary section
            total_time = sum(s["total_ms"] for s in stats.values())
            f.write("\n" + "=" * 70 + "\n")
            f.write("Summary\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total measured time: {total_time:.4f} ms\n")

            # Percentage breakdown
            f.write("\nTime breakdown (%):\n")
            for name, data in sorted_stats:
                pct = (data["total_ms"] / total_time * 100) if total_time > 0 else 0
                f.write(f"  {name}: {pct:.2f}%\n")

        logging.info(f"Saved XVLA timing profile to {self.output_file}")

    def reset(self) -> None:
        with self._lock:
            self.timings.clear()
            self.call_count = 0


# Global profiler instance
_xvla_profiler: XVLATimingProfiler | None = None


def get_xvla_profiler(output_file: str = "xvla_timing_profile.txt") -> XVLATimingProfiler:
    global _xvla_profiler
    if _xvla_profiler is None:
        _xvla_profiler = XVLATimingProfiler(output_file=output_file)
    return _xvla_profiler


class XVLAAttentionCollector:
    """
    Collector that creates attention heatmap visualizations directly during inference.
    Heatmaps are saved at the same interval as timing profiles.
    """

    def __init__(self, output_dir: str = ".", save_interval: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = True  # Enabled by default
        self._sample_counter = 0
        self.save_interval = save_interval
        self._call_count = 0
        self._has_matplotlib = self._check_matplotlib()

    def _check_matplotlib(self) -> bool:
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend for saving
            return True
        except ImportError:
            logging.warning("matplotlib not installed. Attention heatmaps will not be generated.")
            return False

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def set_output_dir(self, output_dir: str) -> None:
        """Set output directory (should match timing profile location)."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _aggregate_heads(self, attention, method: str = "mean"):
        """Aggregate attention weights across heads."""
        if attention.ndim == 4:  # [batch, heads, seq, seq]
            attention = attention[0]  # Take first batch
        if attention.ndim == 3:  # [heads, seq, seq]
            if method == "mean":
                return attention.mean(axis=0)
            elif method == "max":
                return attention.max(axis=0)
        return attention

    def _create_heatmap(
        self,
        attention,
        title: str,
        output_path: Path,
    ) -> None:
        """Create and save a single attention heatmap."""
        if not self._has_matplotlib:
            return

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attention, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Key Position", fontsize=10)
        ax.set_ylabel("Query Position", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _create_layer_comparison(
        self,
        attention_by_layer: dict,
        sample_id: int,
        output_path: Path,
    ) -> None:
        """Create a grid comparing attention across all layers."""
        if not self._has_matplotlib:
            return

        import matplotlib.pyplot as plt
        import numpy as np

        layer_names = sorted(attention_by_layer.keys())
        num_layers = len(layer_names)

        if num_layers == 0:
            return

        cols = min(6, num_layers)
        rows = (num_layers + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))

        if num_layers == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, layer_name in enumerate(layer_names):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            attn = self._aggregate_heads(attention_by_layer[layer_name])
            ax.imshow(attn, cmap="viridis", aspect="auto")
            layer_num = layer_name.replace("policy_layer_", "L")
            ax.set_title(layer_num, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for idx in range(num_layers, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis("off")

        fig.suptitle(f"Sample {sample_id} - Attention Across Layers", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def collect_from_model(self, model: "XVLAModel", sample_id: int | None = None) -> None:
        if not self._enabled:
            logging.debug("Attention collection is disabled, skipping.")
            return

        if not self._has_matplotlib:
            return

        self._call_count += 1

        # Only save at interval (like timing profiler)
        if self.save_interval > 0 and self._call_count % self.save_interval != 0:
            return

        if sample_id is None:
            sample_id = self._sample_counter
            self._sample_counter += 1

        attention_weights = {}
        layers_with_weights = 0

        # Collect from policy transformer blocks
        for layer_idx, block in enumerate(model.transformer.blocks):
            attn_module = block.attn
            weights = attn_module.get_last_attention_weights()
            if weights is not None:
                layers_with_weights += 1
                attention_weights[f"policy_layer_{layer_idx}"] = (
                    weights.detach().cpu().to(torch.float32).numpy()
                )

        if not attention_weights:
            logging.warning(
                f"No attention weights found for sample {sample_id}. "
                f"Make sure _return_attention is True in Attention modules."
            )
            return

        logging.info(f"Creating attention heatmaps for sample {sample_id} ({layers_with_weights} layers)")

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Create layer comparison heatmap (all layers in one image)
        comparison_path = self.output_dir / f"attention_sample_{sample_id}_{timestamp}.png"
        self._create_layer_comparison(attention_weights, sample_id, comparison_path)
        logging.info(f"Saved attention heatmap to {comparison_path}")

        # Optionally create individual layer heatmaps for first, middle, last layers
        key_layers = self._get_key_layers(list(attention_weights.keys()))
        for layer_name in key_layers:
            attn = self._aggregate_heads(attention_weights[layer_name])
            layer_path = self.output_dir / f"attention_sample_{sample_id}_{layer_name}_{timestamp}.png"
            self._create_heatmap(
                attn,
                f"Sample {sample_id} - {layer_name}",
                layer_path,
            )

    def _get_key_layers(self, layer_names: list) -> list:
        """Get first, middle, and last layer names."""
        if len(layer_names) <= 3:
            return layer_names
        sorted_names = sorted(layer_names)
        return [sorted_names[0], sorted_names[len(sorted_names) // 2], sorted_names[-1]]

    def save(self, filename: str | None = None) -> str:
        """Legacy method - now heatmaps are saved directly during collection."""
        logging.info("Attention heatmaps are saved directly during collection, no separate save needed.")
        return str(self.output_dir)

    def reset(self) -> None:
        self._sample_counter = 0
        self._call_count = 0


# Global attention collector instance
_attention_collector: XVLAAttentionCollector | None = None


def get_attention_collector(output_dir: str = ".") -> XVLAAttentionCollector:
    global _attention_collector
    if _attention_collector is None:
        _attention_collector = XVLAAttentionCollector(output_dir=output_dir)
    return _attention_collector


def set_xvla_profiler_output(output_file: str) -> None:
    profiler = get_xvla_profiler()
    profiler.output_file = output_file


class XVLAModel(nn.Module):
    """
    XVLA backbone that stitches Florence-2 embeddings with the temporal/action transformer head.
    """

    def __init__(
        self,
        config: XVLAConfig,
        florence_config: Florence2Config,
        proprio_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size: int = config.chunk_size
        self.use_proprio: bool = config.use_proprio

        # Build action space with auto-detection for "auto" mode
        if config.action_mode.lower() == "auto":
            # Auto-detect real action dim from config.action_feature
            real_dim = (
                config.action_feature.shape[-1]
                if config.action_feature is not None
                else config.max_action_dim
            )
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=real_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())

        self.dim_action = self.action_space.dim_action
        self.dim_proprio = proprio_dim

        self.vlm = Florence2ForConditionalGeneration(florence_config)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=self.dim_action,
            dim_propio=self.dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        # Apply freezing based on config
        self._apply_freezing()

        # Apply dtype casting based on config
        self._apply_dtype()

    def _get_target_dtype(self) -> torch.dtype:
        """Get the target dtype based on config."""
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def _apply_dtype(self) -> None:
        """
        Apply dtype casting to model components based on config.
        """
        target_dtype = self._get_target_dtype()
        self.to(dtype=target_dtype)

    def _apply_freezing(self) -> None:
        """
        Freeze VLM vision and language encoders based on config options.
        Keep only policy transformer and soft prompts trainable.
        """
        # Freeze vision encoder
        if self.config.freeze_vision_encoder and hasattr(self.vlm, "vision_tower"):
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = False

        # Freeze language encoder
        if self.config.freeze_language_encoder and hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            # Freeze encoder
            if hasattr(lm, "model") and hasattr(lm.model, "encoder"):
                for param in lm.model.encoder.parameters():
                    param.requires_grad = False
            # Freeze shared embeddings
            if hasattr(lm, "model") and hasattr(lm.model, "shared"):
                for param in lm.model.shared.parameters():
                    param.requires_grad = False

        # Freeze or unfreeze policy transformer
        if not self.config.train_policy_transformer:
            for name, param in self.transformer.named_parameters():
                if "soft_prompts" not in name:
                    param.requires_grad = False

        # Freeze or unfreeze soft prompts
        if not self.config.train_soft_prompts and hasattr(self.transformer, "soft_prompt_hub"):
            for param in self.transformer.soft_prompt_hub.parameters():
                param.requires_grad = False

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Encode text and multi-view images via Florence2 encoder.
        """
        profiler = get_xvla_profiler()
        t_vlm_total = profiler.start_timer()

        batch_size, num_views = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = pixel_values.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]

        # Time vision encoding
        t_vision = profiler.start_timer()
        valid_feats = self.vlm._encode_image(valid_images)
        profiler.end_timer(t_vision, "vlm_vision_encoder")

        tokens_per_view, hidden_dim = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)

        # Time text embedding
        t_text_embed = profiler.start_timer()
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        profiler.end_timer(t_text_embed, "vlm_text_embedding")

        # Time multimodal merge
        t_merge = profiler.start_timer()
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )
        profiler.end_timer(t_merge, "vlm_multimodal_merge")

        # Time language encoder
        t_lang_enc = profiler.start_timer()
        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]
        profiler.end_timer(t_lang_enc, "vlm_language_encoder")

        aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)

        profiler.end_timer(t_vlm_total, "forward_vlm_total")
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for the XVLA model.
        """
        profiler = get_xvla_profiler()
        t_forward_total = profiler.start_timer()

        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)

        # VLM encoding (detailed timing inside forward_vlm)
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        t = (
            torch.rand(1, device=input_ids.device, dtype=target_dtype)
            + torch.arange(batch_size, device=input_ids.device, dtype=target_dtype) / batch_size
        ) % (1 - 1e-5)

        # Time noise addition and preprocessing
        t_preprocess = profiler.start_timer()
        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)
        profiler.end_timer(t_preprocess, "action_preprocessing")

        # Time transformer forward
        t_transformer = profiler.start_timer()
        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        profiler.end_timer(t_transformer, "policy_transformer_forward")

        # Time loss computation
        t_loss = profiler.start_timer()
        loss = self.action_space.compute_loss(pred_action, action)
        profiler.end_timer(t_loss, "loss_computation")

        profiler.end_timer(t_forward_total, "forward_total")
        return loss

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        self.eval()

        profiler = get_xvla_profiler()
        t_generate_total = profiler.start_timer()

        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        # VLM encoding (detailed timing inside forward_vlm)
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        batch_size = input_ids.shape[0]
        action_dim = self.dim_action

        x1 = torch.randn(batch_size, self.chunk_size, action_dim, device=proprio.device, dtype=target_dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))

        # Time denoising loop
        t_denoising_total = profiler.start_timer()
        denoising_step_times = []

        for i in range(steps, 0, -1):
            t_step = profiler.start_timer()

            t = torch.full((batch_size,), i / steps, device=proprio.device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )

            step_time = profiler.end_timer(t_step, f"denoising_step_{steps - i + 1}")
            denoising_step_times.append(step_time)

        profiler.end_timer(t_denoising_total, "denoising_loop_total")

        # Record average denoising step time
        if denoising_step_times:
            avg_step_time = sum(denoising_step_times) / len(denoising_step_times)
            profiler.record("denoising_step_avg", avg_step_time)

        # Time postprocessing
        t_postprocess = profiler.start_timer()
        result = self.action_space.postprocess(action)
        profiler.end_timer(t_postprocess, "action_postprocessing")

        profiler.end_timer(t_generate_total, "generate_actions_total")
        return result


class XVLAPolicy(PreTrainedPolicy):
    """LeRobot-compliant wrapper built around the XVLA model."""

    config_class = XVLAConfig
    name = "xvla"

    def __init__(self, config: XVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = XVLAModel(config=config, florence_config=florence_config, proprio_dim=proprio_dim)
        self.reset()

    def reset(self) -> None:
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def save_timing_profile(self, output_file: str | None = None) -> None:
        profiler = get_xvla_profiler()
        if output_file is not None:
            profiler.output_file = output_file
            # Also update attention collector to use same directory
            collector = get_attention_collector()
            collector.set_output_dir(str(Path(output_file).parent))
        profiler.save_to_file()

    def reset_timing_profile(self) -> None:
        profiler = get_xvla_profiler()
        profiler.reset()

    def get_timing_stats(self) -> dict[str, dict[str, float]]:
        profiler = get_xvla_profiler()
        return profiler.get_stats()

    def enable_attention_collection(self, output_dir: str | None = None) -> None:
        """
        Enable attention heatmap collection.

        Args:
            output_dir: Directory to save heatmaps. If None, uses same directory
                       as timing profile output file.
        """
        if output_dir is None:
            # Use same directory as timing profiler
            profiler = get_xvla_profiler()
            output_dir = str(Path(profiler.output_file).parent)

        collector = get_attention_collector(output_dir)
        collector.set_output_dir(output_dir)
        collector.enable()
        for block in self.model.transformer.blocks:
            block.attn.set_return_attention(True)

    def disable_attention_collection(self) -> None:
        collector = get_attention_collector()
        collector.disable()
        for block in self.model.transformer.blocks:
            block.attn.set_return_attention(False)

    def collect_attention_weights(self, sample_id: int | None = None) -> None:
        collector = get_attention_collector()
        collector.collect_from_model(self.model, sample_id)

    def save_attention_data(self, filename: str | None = None) -> str:
        collector = get_attention_collector()
        return collector.save(filename)

    def reset_attention_collection(self) -> None:
        collector = get_attention_collector()
        collector.reset()

    def get_attention_collector(self) -> XVLAAttentionCollector:
        return get_attention_collector()

    def get_optim_params(self) -> dict:
        """Return trainable named parameters for optimization.

        Returns a dict of name -> param for all trainable parameters.
        This enables the xvla-adamw optimizer to apply differential learning rates
        based on parameter names (e.g., 1/10 LR for VLM components).
        """
        return dict(filter(lambda kv: kv[1].requires_grad, self.named_parameters()))

    def _prepare_state(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        if not self.config.use_proprio or OBS_STATE not in batch:
            return torch.zeros(batch_size, 0, device=device)
        state = batch[OBS_STATE]
        if state.ndim > 2:
            state = state[:, -1, :]
        return pad_vector(state, self.model.dim_proprio)

    def _prepare_images(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        present_img_keys = [key for key in self.config.image_features if key in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                "All image features are missing from the batch. "
                f"Batch keys: {list(batch.keys())}, expected at least one of {list(self.config.image_features)}."
            )

        images = []
        masks = []
        for key in present_img_keys:
            img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding)
            images.append(img)
            masks.append(torch.ones(img.size(0), dtype=torch.bool, device=img.device))

        stacked_imgs = torch.stack(images, dim=1)
        stacked_masks = torch.stack(masks, dim=1)

        total_views = self.config.num_image_views or stacked_imgs.size(1)
        total_views = max(total_views, stacked_imgs.size(1))
        num_pad = total_views - stacked_imgs.size(1)
        if num_pad > 0:
            pad_shape = (stacked_imgs.size(0), num_pad, *stacked_imgs.shape[2:])
            pad_imgs = stacked_imgs.new_zeros(pad_shape)
            pad_masks = stacked_masks.new_zeros((stacked_masks.size(0), num_pad))
            stacked_imgs = torch.cat([stacked_imgs, pad_imgs], dim=1)
            stacked_masks = torch.cat([stacked_masks, pad_masks], dim=1)

        return stacked_imgs, stacked_masks

    def _get_domain_id(self, batch: dict[str, Tensor], batch_size: int, device: torch.device) -> Tensor:
        candidate = None
        if self.config.domain_feature_key and self.config.domain_feature_key in batch:
            candidate = batch[self.config.domain_feature_key]
        elif "domain_id" in batch:
            candidate = batch["domain_id"]

        if candidate is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        if not isinstance(candidate, torch.Tensor):
            candidate = torch.as_tensor(candidate, device=device)
        else:
            candidate = candidate.to(device=device)

        if candidate.ndim == 0:
            candidate = candidate.expand(batch_size)
        if candidate.ndim > 1:
            candidate = candidate.view(candidate.shape[0], -1)[:, 0]
        if candidate.shape[0] != batch_size:
            candidate = candidate.expand(batch_size)
        return candidate.to(dtype=torch.long)

    def _prepare_action_targets(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION not in batch:
            raise ValueError("Batch is missing action targets required for training.")
        actions = batch[ACTION]
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = pad_tensor_along_dim(actions, self.config.chunk_size, dim=1)
        if actions.shape[-1] != self.model.dim_action:
            actions = pad_vector(actions, self.model.dim_action)
        return actions

    def _build_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        profiler = get_xvla_profiler()
        t_build = profiler.start_timer()

        input_ids = batch[OBS_LANGUAGE_TOKENS]
        batch_size = input_ids.shape[0]
        images, image_mask = self._prepare_images(batch)
        domain_id = self._get_domain_id(batch, batch_size, images.device)
        proprio = self._prepare_state(batch, batch_size, images.device)

        profiler.end_timer(t_build, "batch_preparation")
        return {
            "input_ids": input_ids,
            "image_input": images,
            "image_mask": image_mask,
            "domain_id": domain_id,
            "proprio": proprio,
        }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        profiler = get_xvla_profiler()
        t_policy_forward = profiler.start_timer()

        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch)
        losses = self.model(action=targets, **inputs)
        total_loss = sum(losses.values())

        log_dict = {k: v.detach().item() for k, v in losses.items()}
        log_dict["loss"] = total_loss.detach().item()

        profiler.end_timer(t_policy_forward, "policy_forward_total")
        return total_loss, log_dict

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        profiler = get_xvla_profiler()
        t_action_chunk = profiler.start_timer()

        inputs = self._build_model_inputs(batch)
        actions = self.model.generate_actions(**inputs, steps=self.config.num_denoising_steps)

        # Automatically collect attention weights if enabled
        collector = get_attention_collector()
        if collector.is_enabled():
            collector.collect_from_model(self.model)

        profiler.end_timer(t_action_chunk, "get_action_chunk_total")
        return actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        profiler = get_xvla_profiler()
        t_predict = profiler.start_timer()

        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        result = self._get_action_chunk(batch)

        profiler.end_timer(t_predict, "predict_action_chunk_total")
        return result

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        profiler = get_xvla_profiler()
        t_select = profiler.start_timer()

        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            t_generate = profiler.start_timer()
            actions = self._get_action_chunk(batch)
            profiler.end_timer(t_generate, "action_generation_in_select")
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        result = self._queues[ACTION].popleft()
        profiler.end_timer(t_select, "select_action_total")
        return result

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        Loads XVLA model weights with:
        - automatic prefix 'model.' added to all keys
        - skip list for layers that should remain randomly initialized
        """
        import safetensors.torch

        # step 1: load config
        # TODO: jadechoghari, fix this
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        # step 2: locate model.safetensors
        if os.path.isdir(model_id):
            logging.info("Loading weights from local directory")
            model_file = os.path.join(model_id, "model.safetensors")
        else:
            try:
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import HfHubHTTPError

                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(f"model.safetensors not found on the Hub at {model_id}") from e

        logging.info(f"Loading checkpoint from {model_file}")
        # step 3: load state dict
        state_dict = safetensors.torch.load_file(model_file)
        encoder_key = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        shared_key = "model.vlm.language_model.model.shared.weight"
        if encoder_key in state_dict:
            state_dict[shared_key] = state_dict[encoder_key]
            # or deepcopy
        # step 4: load into instance
        instance.load_state_dict(state_dict, strict=True)
        logging.info("Loaded XVLA checkpoint")
        # step 5: finalize
        # Reapply dtype after loading state dict
        instance.model._apply_dtype()
        instance.to(config.device)
        instance.eval()
        return instance


def resize_with_pad(img: torch.Tensor, height: int, width: int, pad_value: float = 0.0) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] == new_dim:
        return vector
    if new_dim == 0:
        shape = list(vector.shape)
        shape[-1] = 0
        return vector.new_zeros(*shape)
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = vector.new_zeros(*shape)
    length = min(current_dim, new_dim)
    new_vector[..., :length] = vector[..., :length]
    return new_vector


def pad_tensor_along_dim(tensor: Tensor, target_len: int, dim: int = 1) -> Tensor:
    current_len = tensor.size(dim)
    if current_len == target_len:
        return tensor
    if current_len > target_len:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, target_len)
        return tensor[tuple(slices)]
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - current_len
    pad_tensor = tensor.new_zeros(pad_shape)
    return torch.cat([tensor, pad_tensor], dim=dim)
