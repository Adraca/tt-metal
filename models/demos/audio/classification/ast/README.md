# Audio Spectrogram Transformer (AST) — TTNN Bring-Up

> **Bounty**: [tenstorrent/tt-metal#52054](https://github.com/tenstorrent/tt-metal/issues/52054)

TTNN implementation of the **Audio Spectrogram Transformer (AST)** for AudioSet
classification. The AST adapts the Vision Transformer (ViT) architecture to
audio by treating mel-spectrograms as images, extracting overlapping patches,
and processing them through a standard transformer encoder.

## Model Architecture

| Parameter               | Value                                  |
|-------------------------|----------------------------------------|
| **Checkpoint**          | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| **Base architecture**   | Vision Transformer (ViT-B/16), pre-LayerNorm |
| **Hidden size**         | 768                                    |
| **Encoder layers**      | 12                                     |
| **Attention heads**     | 12                                     |
| **Head dimension**      | 64                                     |
| **FFN intermediate**    | 3072                                   |
| **Patch size**          | 16 × 16                                |
| **Time stride**         | 10                                     |
| **Frequency stride**    | 10                                     |
| **Activation**          | GELU                                   |
| **LayerNorm epsilon**   | 1e-12                                  |
| **Output classes**      | 527 (AudioSet)                         |
| **Parameters**          | ~87M                                   |

## Input Pipeline

```
Raw Audio
  │
  ▼
Mel-Spectrogram: [batch, 1024, 128]    ← 1024 time frames × 128 mel bins
  │
  ▼
Patch Extraction (unfold):
  stride = (10, 10), patch = 16×16
  num_time_patches  = (1024 - 16) / 10 + 1 = 101
  num_freq_patches  = (128  - 16) / 10 + 1 = 12
  total patches     = 101 × 12 = 1212
  │
  ▼
Flattened Patches: [batch, 1212, 256]   ← each patch is 16×16 = 256-d
  │
  ▼
Linear Projection: [batch, 1212, 768]
  │
  ▼
Prepend Tokens: [CLS] + [DIST] + patches → [batch, 1214, 768]
  │
  ▼
Add Positional Embeddings: [batch, 1214, 768]
  │
  ▼
12× Encoder Blocks (pre-LN ViT):
  LayerNorm → MHSA → Residual → LayerNorm → FFN → Residual
  │
  ▼
Final LayerNorm → Pool (mean of CLS + DIST) → Classifier → [batch, 527]
```

## Project Structure

```
tenstorrent_ast_bringup/
├── README.md                  ← This file
├── demo/
│   └── demo.py                ← Interactive demo with PCC and inference modes
├── models/
│   └── ast_model.py           ← TTNN AST model implementation
└── tests/
    └── test_ast.py            ← CI-compatible pytest suite
```

## Quick Start

### Run Demo

```bash
# PCC validation against PyTorch reference
python demo/demo.py --mode pcc

# Inference with a sample audio file
python demo/demo.py --mode inference --audio sample.wav
```

### Run Tests

```bash
# Full test suite
pytest tests/test_ast.py -v

# Individual tests
pytest tests/test_ast.py::test_ast_embedding_pcc -v
pytest tests/test_ast.py::test_ast_single_encoder_block_pcc -v
pytest tests/test_ast.py::test_ast_pcc -v
pytest tests/test_ast.py::test_ast_top1_accuracy -v
```

### Test Descriptions

| Test                              | Description                                         |
|-----------------------------------|-----------------------------------------------------|
| `test_ast_embedding_pcc`          | Patch projection + CLS/DIST prepend + positional embed PCC |
| `test_ast_single_encoder_block_pcc` | Single encoder layer (block 0) PCC against PyTorch  |
| `test_ast_pcc`                    | Full 12-layer model PCC ≥ 0.99                      |
| `test_ast_top1_accuracy`          | Top-1 argmax matches PyTorch reference               |

## Performance Results

> Measured on Tenstorrent Grayskull/Wormhole hardware. Values pending hardware run.

| Metric              | Value         |
|---------------------|---------------|
| **Compile time**    | _pending_     |
| **Warm latency**    | _pending_     |
| **Throughput**      | _pending_     |
| **Batch size**      | 1             |
| **Hardware**        | _pending_     |
| **PCC (full model)**| _pending_     |
| **Top-1 match**     | _pending_     |

## Software Revisions

| Component        | Version / Commit |
|------------------|------------------|
| **tt-metal**     | _pending_        |
| **ttnn**         | _pending_        |
| **PyTorch**      | ≥ 2.0            |
| **Transformers** | ≥ 4.30           |
| **Python**       | ≥ 3.10           |

## Technical Notes

### Tile Alignment
All tensors are padded to multiples of 32 along the last two dimensions before
conversion to `TILE_LAYOUT` on device. The sequence dimension 1214 pads to 1216
and hidden dimension 768 is already tile-aligned (768 = 24 × 32).

### Pre-LayerNorm Architecture
The AST uses **pre-LayerNorm** (LN-before-attention) architecture, matching the
DeiT/ViT convention. This is critical for PCC — using post-LayerNorm will
produce incorrect results.

### Fused QKV
Query, Key, and Value projection weights are fused into a single
`[HIDDEN_SIZE, 3×HIDDEN_SIZE]` matrix for efficient matmul on device. The
output is split on CPU for the head-reshape step.

### Memory Management
Intermediate tensors are explicitly deallocated via `ttnn.deallocate()` after
use to prevent L1 SRAM exhaustion on device. Each encoder block's weights are
loaded, used, and freed before proceeding to the next layer.

## References

- **HuggingFace Checkpoint**: [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
- **AST Paper**: [AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) (Gong et al., 2021)
- **tt-metal ViT Reference**: [tt-metal ViT implementation](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/grayskull/vit)
- **Bounty Issue**: [tenstorrent/tt-metal#52054](https://github.com/tenstorrent/tt-metal/issues/52054)
- **DeiT Paper**: [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877) (Touvron et al., 2021)
