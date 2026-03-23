#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Convert DeepSeek expert weights to a BSPM-pre-quantized tt-metal weight cache.

Runs on the target hardware (TG / T3K / N300) with a live mesh device.  For
each MoE layer the script applies BitSculpt BSPM tile assignments to the
dequantized expert weights before the standard bfloat4_b/bfloat8_b conversion,
producing a weight cache that reflects the mixed-precision allocation.

Usage
-----
    # Minimal — TG system, R1-0528, Variant B 3.5 b/e:
    MESH_DEVICE=TG \\
    python models/demos/deepseek_v3/scripts/convert_bspm_weights.py \\
        --model-path  /proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528 \\
        --output-path /localdev/mtairum/deepseek_cache/bspm_B_3.5 \\
        --bspm-dir    /localdev/mtairum/bit_sculpt/results \\
        --bspm-model  deepseek-r1-0528

    # Custom variant / budget:
        --bspm-variant A --bspm-budget 4.0

    # Dry-run: preprocessing only, no device conversion (fast sanity check):
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig

# ---------------------------------------------------------------------------
# Helpers shared with test_bspm_demo.py
# ---------------------------------------------------------------------------

_TT_CODE_TO_MANT: dict[int, int | None] = {0: 7, 1: 3, 2: 1, 3: None}
_PROJ_IDX = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
_EXPERT_WEIGHT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def _apply_bspm_to_weight_tensor(
    tensor: torch.Tensor,
    codes_1d: np.ndarray,
    quantize_dequantize_bfp,
) -> torch.Tensor:
    """Apply one expert's BSPM codes lazily to a single HF weight tensor."""
    orig_dtype = tensor.dtype
    weight_kn = tensor.float().numpy().T  # HF stores (N, K); BSPM codes use (K, N)
    k_dim, n_dim = weight_kn.shape
    tiles_h = k_dim // 32
    tiles_w = n_dim // 32
    expected_tiles = tiles_h * tiles_w

    if codes_1d.shape[0] < expected_tiles:
        raise ValueError(f"BSPM codes have {codes_1d.shape[0]} tiles, expected at least {expected_tiles}")

    codes_2d = codes_1d[:expected_tiles].reshape(tiles_h, tiles_w)
    unique_codes = np.unique(codes_2d)
    non_bfp4 = unique_codes[unique_codes != 1]
    if len(non_bfp4) == 0:
        return tensor

    weight_tiled = weight_kn.reshape(tiles_h, 32, tiles_w, 32).transpose(0, 2, 1, 3).copy()
    for code in non_bfp4:
        mant_bits = _TT_CODE_TO_MANT.get(int(code), 3)
        tile_mask = codes_2d == code
        if mant_bits is None:
            weight_tiled[tile_mask] = 0.0
        else:
            weight_tiled[tile_mask] = quantize_dequantize_bfp(weight_tiled[tile_mask], mant_bits)

    weight_out = weight_tiled.transpose(0, 2, 1, 3).reshape(k_dim, n_dim).T
    return torch.from_numpy(weight_out.copy()).to(orig_dtype)


class _BSPMStateDict:
    """Mapping wrapper that applies BSPM preprocessing only when an expert
    weight is actually requested."""

    def __init__(
        self,
        base,
        hf_config,
        bspm_dir: Path,
        bspm_model: str,
        variant: str,
        budget: float,
        bspm_root: Path,
        *,
        full_prefix: str = "",
        layer_codes_cache: dict[int, np.ndarray | None] | None = None,
        missing_layers: set[int] | None = None,
    ):
        self._base = base
        self._hf_config = hf_config
        self._bspm_dir = bspm_dir
        self._bspm_model = bspm_model
        self._variant = variant
        self._budget = budget
        self._full_prefix = full_prefix
        self._layer_codes_cache = {} if layer_codes_cache is None else layer_codes_cache
        self._missing_layers = set() if missing_layers is None else missing_layers

        bspm_root_str = str(bspm_root)
        if bspm_root_str not in sys.path:
            sys.path.insert(0, bspm_root_str)

        from integration.ttnn.bspm_loader import load_bspm_for_layer

        from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import quantize_dequantize_bfp

        self._load_bspm_for_layer = load_bspm_for_layer
        self._quantize_dequantize_bfp = quantize_dequantize_bfp

    def _get_layer_codes(self, layer_idx: int) -> np.ndarray | None:
        if layer_idx not in self._layer_codes_cache:
            bspm_file = (
                self._bspm_dir
                / self._bspm_model
                / f"layer_{layer_idx}"
                / "precision_eval"
                / f"precision_map_{self._variant}_{self._budget:.1f}.bspm"
            )
            if not bspm_file.exists():
                if layer_idx not in self._missing_layers:
                    logger.warning(f"Layer {layer_idx}: BSPM file not found, using raw weights — {bspm_file}")
                    self._missing_layers.add(layer_idx)
                self._layer_codes_cache[layer_idx] = None
            else:
                self._layer_codes_cache[layer_idx] = self._load_bspm_for_layer(str(bspm_file))["codes"]
        return self._layer_codes_cache[layer_idx]

    def __getitem__(self, key):
        full_key = f"{self._full_prefix}{key}"
        tensor = self._base[key]
        match = _EXPERT_WEIGHT_RE.match(full_key)
        if match is None:
            return tensor

        layer_idx = int(match.group(1))
        expert_idx = int(match.group(2))
        proj_name = match.group(3)

        layer_codes = self._get_layer_codes(layer_idx)
        if layer_codes is None or expert_idx >= layer_codes.shape[0]:
            return tensor

        return _apply_bspm_to_weight_tensor(
            tensor,
            layer_codes[expert_idx, _PROJ_IDX[proj_name]],
            self._quantize_dequantize_bfp,
        )

    def __contains__(self, key):
        return key in self._base

    def __iter__(self):
        return iter(self._base)

    def __len__(self):
        return len(self._base)

    def keys(self):
        return self._base.keys()

    def items(self):
        for k in self._base:
            yield k, self[k]

    def values(self):
        for k in self._base:
            yield self[k]

    def view_with_prefix(self, prefix: str, num_layers: int | None = None) -> "_BSPMStateDict":
        return _BSPMStateDict(
            self._base.view_with_prefix(prefix, num_layers),
            self._hf_config,
            self._bspm_dir,
            self._bspm_model,
            self._variant,
            self._budget,
            self._bspm_dir.parent,
            full_prefix=f"{self._full_prefix}{prefix}",
            layer_codes_cache=self._layer_codes_cache,
            missing_layers=self._missing_layers,
        )

    def clear_cache(self) -> None:
        clear_cache = getattr(self._base, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()

    def evict(self, key: str) -> None:
        evict = getattr(self._base, "evict", None)
        if callable(evict):
            evict(key)


def _dry_run_bspm_preprocessing(state_dict, hf_config) -> None:
    first_k_dense = getattr(hf_config, "first_k_dense_replace", 3)
    evict = getattr(state_dict, "evict", None)
    clear_cache = getattr(state_dict, "clear_cache", None)

    for layer_idx in tqdm(
        range(first_k_dense, hf_config.num_hidden_layers),
        desc="BSPM preprocessing layers",
        unit="layer",
        total=hf_config.num_hidden_layers - first_k_dense,
    ):
        for expert_idx in range(hf_config.n_routed_experts):
            for proj_name in _PROJ_IDX:
                key = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.weight"
                if key not in state_dict:
                    continue
                _ = state_dict[key]
                if callable(evict):
                    evict(key)
        if callable(clear_cache):
            clear_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek expert weights to a BSPM-pre-quantized tt-metal weight cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the HF model dir (FP8 or dequantized).")
    parser.add_argument("--output-path", type=Path, required=True, help="Output weight cache directory.")
    parser.add_argument(
        "--bspm-dir", type=Path, required=True, help="BitSculpt results root (contains <bspm-model>/ subdirs)."
    )
    parser.add_argument(
        "--bspm-model", type=str, required=True, help="BSPM model sub-directory, e.g. deepseek-r1-0528."
    )
    parser.add_argument("--bspm-variant", type=str, default="B", help="BSPM variant letter.")
    parser.add_argument("--bspm-budget", type=float, default=3.5, help="Bits-per-element budget.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run BSPM preprocessing only (no device / no conversion). Useful for timing and sanity checks.",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()

    # ── Resolve dequantized model path ──────────────────────────────────────
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict

    deq_path = args.model_path
    if not deq_path.exists():
        logger.error(
            f"Dequantized checkpoint not found at {deq_path}. " f"Run scripts/dequantize_hf_checkpoint.py first."
        )
        sys.exit(1)

    logger.info(f"Loading dequantized weights from {deq_path}")
    state_dict = LazyStateDict(deq_path)

    # ── HF config ───────────────────────────────────────────────────────────
    hf_config = AutoConfig.from_pretrained(args.model_path.resolve(), trust_remote_code=True)
    logger.info(f"Model: {hf_config.num_hidden_layers} layers, {hf_config.n_routed_experts} experts/layer")

    # ── Derive bspm_root from bspm_dir ──────────────────────────────────────
    bspm_root = args.bspm_dir.parent

    # ── BSPM preprocessing wrapper ──────────────────────────────────────────
    logger.info(
        f"Applying lazy BSPM pre-quantization: variant={args.bspm_variant}, budget={args.bspm_budget} b/e, "
        f"layers {getattr(hf_config, 'first_k_dense_replace', 3)}–{hf_config.num_hidden_layers - 1}. "
        "Expert weights will be transformed on demand during conversion."
    )
    bspm_state_dict = _BSPMStateDict(
        state_dict,
        hf_config,
        args.bspm_dir,
        args.bspm_model,
        args.bspm_variant,
        args.bspm_budget,
        bspm_root,
    )

    if args.dry_run:
        t0 = time.time()
        _dry_run_bspm_preprocessing(bspm_state_dict, hf_config)
        logger.info(f"BSPM dry-run preprocessing done in {time.time() - t0:.1f}s")
        return

    # ── Device setup ────────────────────────────────────────────────────────
    import ttnn
    from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel, get_fabric_config
    from models.demos.deepseek_v3.utils.weight_config import get_weight_config

    mesh_shape_env = os.environ.get("MESH_DEVICE", "TG")
    from models.demos.deepseek_v3.utils.test_utils import SYSTEM_NAME_TO_MESH_SHAPE

    mesh_shape = SYSTEM_NAME_TO_MESH_SHAPE.get(mesh_shape_env, (4, 8))
    logger.info(f"Opening mesh device {mesh_shape[0]}×{mesh_shape[1]} ({mesh_shape_env})")

    fabric_config = get_fabric_config()
    if fabric_config:
        ttnn.set_fabric_config(fabric_config)

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
    )

    try:
        output_path = args.output_path
        if output_path.exists() and not args.force:
            logger.error(f"Output path {output_path} already exists. Use --force to overwrite.")
            sys.exit(1)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting weights → {output_path}")
        t1 = time.time()
        get_weight_config(
            RowBatchedModel,
            hf_config,
            (bspm_state_dict,),
            output_path,
            mesh_device,
            force_recalculate=True,
        )
        logger.info(f"Weight conversion done in {time.time() - t1:.1f}s")
        logger.info(f"BSPM weight cache written to {output_path}")
    finally:
        ttnn.close_mesh_device(mesh_device)
        if fabric_config:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
