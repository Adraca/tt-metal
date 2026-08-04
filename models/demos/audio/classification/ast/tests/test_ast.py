# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CI-compatible pytest suite for the Audio Spectrogram Transformer (AST) bring-up.

Bounty reference: tenstorrent/tt-metal#52054
Checkpoint: MIT/ast-finetuned-audioset-10-10-0.4593

Tests validate TTNN model outputs against PyTorch reference using Pearson
Correlation Coefficient (PCC). The AST processes audio spectrograms
(1024 time frames × 128 mel bins) by unfolding them into 1212 overlapping
patches of size 16×16, then prepending CLS and distillation tokens to form
a 1214-token sequence through a 12-layer ViT encoder.
"""

import math

import pytest
import torch
import ttnn
from transformers import ASTConfig, ASTModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
EXPECTED_PCC = 0.99
TIME_FRAMES = 1024
MEL_BINS = 128
PATCH_SIZE = 16
FREQUENCY_STRIDE = 10
TIME_STRIDE = 10
NUM_FREQUENCY_PATCHES = (MEL_BINS - PATCH_SIZE) // FREQUENCY_STRIDE + 1  # 12
NUM_TIME_PATCHES = (TIME_FRAMES - PATCH_SIZE) // TIME_STRIDE + 1  # 101
NUM_PATCHES = NUM_FREQUENCY_PATCHES * NUM_TIME_PATCHES  # 1212
PATCH_DIM = PATCH_SIZE * PATCH_SIZE  # 256
NUM_TOKENS = NUM_PATCHES + 2  # 1214 (CLS + distillation)
HIDDEN_SIZE = 768
NUM_ENCODER_LAYERS = 12
NUM_ATTENTION_HEADS = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_pcc(golden: torch.Tensor, predicted: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors.

    Handles edge cases where one or both tensors have zero variance,
    which would otherwise produce NaN from ``torch.corrcoef``.

    Args:
        golden: Reference tensor (any shape, will be flattened).
        predicted: Predicted tensor (same total numel as *golden*).

    Returns:
        PCC value in ``[-1.0, 1.0]``. Returns ``1.0`` when both tensors
        are constant and identical, ``0.0`` when only one has zero variance.
    """
    golden_flat = golden.float().flatten()
    predicted_flat = predicted.float().flatten()
    if golden_flat.std() == 0 and predicted_flat.std() == 0:
        return 1.0
    if golden_flat.std() == 0 or predicted_flat.std() == 0:
        return 0.0
    return torch.corrcoef(torch.stack([golden_flat, predicted_flat]))[0, 1].item()


def _load_reference_model() -> ASTModel:
    """Load the pretrained AST model in eval mode on CPU."""
    model = ASTModel.from_pretrained(MODEL_NAME)
    model.eval()
    return model


def _create_dummy_spectrogram(batch_size: int) -> torch.Tensor:
    """Create a reproducible dummy spectrogram input.

    Shape: ``[batch_size, TIME_FRAMES, MEL_BINS]`` — raw mel-spectrogram
    before patch extraction.
    """
    torch.manual_seed(42)
    return torch.randn(batch_size, TIME_FRAMES, MEL_BINS)


def _unfold_spectrogram(spectrogram: torch.Tensor) -> torch.Tensor:
    """Unfold a mel-spectrogram into flattened 16×16 patches on CPU.

    Mimics the ``ASTModel.embeddings`` patch extraction with stride
    ``(TIME_STRIDE, FREQUENCY_STRIDE)``.

    Args:
        spectrogram: ``[batch, TIME_FRAMES, MEL_BINS]``

    Returns:
        Patch tensor ``[batch, NUM_PATCHES, PATCH_DIM]`` (1212 patches of 256-d).
    """
    batch_size = spectrogram.shape[0]
    # Reshape to image-like: [B, 1, T, F]
    x = spectrogram.unsqueeze(1)
    # unfold time dimension
    x = x.unfold(2, PATCH_SIZE, TIME_STRIDE)  # [B, 1, num_t, F, patch_h]
    # unfold frequency dimension
    x = x.unfold(3, PATCH_SIZE, FREQUENCY_STRIDE)  # [B, 1, num_t, num_f, patch_h, patch_w]
    # Reshape to [B, num_patches, patch_dim]
    x = x.contiguous().view(batch_size, -1, PATCH_DIM)
    assert x.shape == (batch_size, NUM_PATCHES, PATCH_DIM), (
        f"Unexpected patch shape {x.shape}, expected ({batch_size}, {NUM_PATCHES}, {PATCH_DIM})"
    )
    return x


def _get_reference_embeddings(model: ASTModel, spectrogram: torch.Tensor) -> torch.Tensor:
    """Run the PyTorch AST embedding layer and return hidden states.

    The AST embedding layer projects patches, prepends CLS + distillation
    tokens, adds positional embeddings, and applies LayerNorm + dropout.

    Args:
        model: Pretrained ``ASTModel``.
        spectrogram: ``[batch, TIME_FRAMES, MEL_BINS]``.

    Returns:
        Embedding output ``[batch, NUM_TOKENS, HIDDEN_SIZE]``.
    """
    with torch.no_grad():
        # ASTModel expects input_values: [batch, time, freq]
        outputs = model.embeddings(spectrogram)
    return outputs


def _get_reference_encoder_block_output(
    model: ASTModel,
    hidden_states: torch.Tensor,
    layer_idx: int = 0,
) -> torch.Tensor:
    """Run a single PyTorch encoder layer and return its output.

    Args:
        model: Pretrained ``ASTModel``.
        hidden_states: ``[batch, seq_len, HIDDEN_SIZE]``.
        layer_idx: Which encoder layer to run (0-indexed).

    Returns:
        Encoder block output ``[batch, seq_len, HIDDEN_SIZE]``.
    """
    encoder_layer = model.encoder.layer[layer_idx]
    with torch.no_grad():
        # ASTEncoder layers return tuple (hidden_states, attention_weights)
        layer_output = encoder_layer(hidden_states, head_mask=None)
    return layer_output[0]


def _pad_to_tile(tensor: torch.Tensor) -> torch.Tensor:
    """Pad the last two dims of *tensor* to multiples of 32 for TILE_LAYOUT.

    Args:
        tensor: Input tensor of shape ``[..., H, W]``.

    Returns:
        Padded tensor where both H and W are multiples of 32.
    """
    h, w = tensor.shape[-2], tensor.shape[-1]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h == 0 and pad_w == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))


def _to_ttnn_tile(tensor: torch.Tensor, device: ttnn.Device) -> ttnn.Tensor:
    """Convert a CPU torch.Tensor to a ttnn TILE_LAYOUT tensor on device.

    Pads last two dims to tile-alignment (multiples of 32), converts to
    bfloat16, and moves to the specified device.

    Args:
        tensor: CPU float32 tensor.
        device: Target ttnn device.

    Returns:
        ``ttnn.Tensor`` in TILE_LAYOUT on *device*.
    """
    padded = _pad_to_tile(tensor)
    tt_tensor = ttnn.from_torch(padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_tensor = ttnn.to_device(tt_tensor, device)
    return tt_tensor


def _from_ttnn(tt_tensor: ttnn.Tensor, target_shape: tuple) -> torch.Tensor:
    """Convert a ttnn tensor back to a CPU torch.Tensor, removing tile padding.

    Args:
        tt_tensor: Device-side ttnn tensor.
        target_shape: Desired output shape (without tile padding).

    Returns:
        CPU float32 tensor with *target_shape*.
    """
    tt_tensor = ttnn.from_device(tt_tensor)
    tt_tensor = ttnn.to_layout(tt_tensor, ttnn.ROW_MAJOR_LAYOUT)
    result = ttnn.to_torch(tt_tensor).float()
    # Slice away tile padding
    slices = [slice(0, s) for s in target_shape]
    return result[tuple(slices)]


# ---------------------------------------------------------------------------
# TTNN Model Components
# ---------------------------------------------------------------------------


def ttnn_ast_patch_embedding(
    patches: ttnn.Tensor,
    weight: ttnn.Tensor,
    bias: ttnn.Tensor,
    cls_token: ttnn.Tensor,
    distillation_token: ttnn.Tensor,
    position_embeddings: ttnn.Tensor,
    device: ttnn.Device,
) -> ttnn.Tensor:
    """TTNN implementation of AST patch embedding + positional encoding.

    Performs: Linear(patches) → Prepend CLS + Distillation → Add PosEmbed

    Args:
        patches: ``[batch, NUM_PATCHES, PATCH_DIM]`` (tile-aligned).
        weight: Projection weight ``[PATCH_DIM, HIDDEN_SIZE]`` (tile-aligned).
        bias: Projection bias ``[1, 1, HIDDEN_SIZE]`` (tile-aligned).
        cls_token: ``[1, 1, HIDDEN_SIZE]`` (tile-aligned).
        distillation_token: ``[1, 1, HIDDEN_SIZE]`` (tile-aligned).
        position_embeddings: ``[1, NUM_TOKENS, HIDDEN_SIZE]`` (tile-aligned).
        device: Target ttnn device.

    Returns:
        ``ttnn.Tensor`` of shape ``[batch, NUM_TOKENS, HIDDEN_SIZE]`` (tile-aligned).
    """
    # Linear projection: [B, 1212, 256] @ [256, 768] → [B, 1212, 768]
    projected = ttnn.matmul(patches, weight)
    projected = ttnn.add(projected, bias)

    # Prepend CLS and distillation tokens along sequence dim
    # [B, 1, 768] + [B, 1, 768] + [B, 1212, 768] → [B, 1214, 768]
    embeddings = ttnn.concat([cls_token, distillation_token, projected], dim=-2)
    ttnn.deallocate(projected)

    # Add positional embeddings
    embeddings = ttnn.add(embeddings, position_embeddings)

    return embeddings


def ttnn_ast_attention(
    hidden_states: ttnn.Tensor,
    qkv_weight: ttnn.Tensor,
    qkv_bias: ttnn.Tensor,
    proj_weight: ttnn.Tensor,
    proj_bias: ttnn.Tensor,
    num_heads: int,
    head_dim: int,
    device: ttnn.Device,
) -> ttnn.Tensor:
    """TTNN multi-head self-attention for a single AST encoder layer.

    Fused QKV projection → split heads → scaled dot-product → project out.

    Args:
        hidden_states: ``[batch, seq_len, HIDDEN_SIZE]``.
        qkv_weight: Fused QKV weight ``[HIDDEN_SIZE, 3*HIDDEN_SIZE]``.
        qkv_bias: Fused QKV bias ``[1, 1, 3*HIDDEN_SIZE]``.
        proj_weight: Output projection weight ``[HIDDEN_SIZE, HIDDEN_SIZE]``.
        proj_bias: Output projection bias ``[1, 1, HIDDEN_SIZE]``.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        device: Target ttnn device.

    Returns:
        Attention output ``[batch, seq_len, HIDDEN_SIZE]``.
    """
    # Fused QKV: [B, S, H] @ [H, 3H] → [B, S, 3H]
    qkv = ttnn.matmul(hidden_states, qkv_weight)
    qkv = ttnn.add(qkv, qkv_bias)

    # Split into Q, K, V each [B, S, H]
    # Using slice operations along the last dimension
    batch = hidden_states.shape[0]
    seq_len = hidden_states.shape[-2]
    hidden_size = num_heads * head_dim

    qkv_torch = _from_ttnn(qkv, (batch, seq_len, 3 * hidden_size))
    ttnn.deallocate(qkv)

    q, k, v = qkv_torch.split(hidden_size, dim=-1)

    # Reshape to multi-head: [B, S, H] → [B, num_heads, S, head_dim]
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    # Move back to device
    q_tt = _to_ttnn_tile(q, device)
    k_tt = _to_ttnn_tile(k, device)
    v_tt = _to_ttnn_tile(v, device)

    # Scaled dot-product attention
    scale = 1.0 / math.sqrt(head_dim)
    k_t = ttnn.transpose(k_tt, -2, -1)
    ttnn.deallocate(k_tt)

    attn_weights = ttnn.matmul(q_tt, k_t)
    ttnn.deallocate(q_tt)
    ttnn.deallocate(k_t)

    attn_weights = ttnn.multiply(attn_weights, scale)
    attn_weights = ttnn.softmax(attn_weights, dim=-1)

    attn_output = ttnn.matmul(attn_weights, v_tt)
    ttnn.deallocate(attn_weights)
    ttnn.deallocate(v_tt)

    # Merge heads: [B, num_heads, S, head_dim] → [B, S, H]
    attn_torch = _from_ttnn(attn_output, (batch, num_heads, seq_len, head_dim))
    ttnn.deallocate(attn_output)

    attn_merged = attn_torch.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
    attn_tt = _to_ttnn_tile(attn_merged, device)

    # Output projection
    output = ttnn.matmul(attn_tt, proj_weight)
    ttnn.deallocate(attn_tt)
    output = ttnn.add(output, proj_bias)

    return output


def ttnn_ast_encoder_block(
    hidden_states: ttnn.Tensor,
    ln1_weight: ttnn.Tensor,
    ln1_bias: ttnn.Tensor,
    qkv_weight: ttnn.Tensor,
    qkv_bias: ttnn.Tensor,
    proj_weight: ttnn.Tensor,
    proj_bias: ttnn.Tensor,
    ln2_weight: ttnn.Tensor,
    ln2_bias: ttnn.Tensor,
    ff_weight1: ttnn.Tensor,
    ff_bias1: ttnn.Tensor,
    ff_weight2: ttnn.Tensor,
    ff_bias2: ttnn.Tensor,
    num_heads: int,
    head_dim: int,
    device: ttnn.Device,
) -> ttnn.Tensor:
    """TTNN implementation of a single AST (pre-LayerNorm ViT) encoder block.

    Architecture: LN → MHSA → Residual → LN → FFN → Residual

    Args:
        hidden_states: ``[batch, seq_len, HIDDEN_SIZE]``.
        ln1_weight, ln1_bias: LayerNorm before attention.
        qkv_weight, qkv_bias: Fused QKV projection weights.
        proj_weight, proj_bias: Attention output projection.
        ln2_weight, ln2_bias: LayerNorm before FFN.
        ff_weight1, ff_bias1: FFN first linear layer.
        ff_weight2, ff_bias2: FFN second linear layer.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        device: Target ttnn device.

    Returns:
        Encoder block output ``[batch, seq_len, HIDDEN_SIZE]``.
    """
    # Pre-LayerNorm 1
    ln1_out = ttnn.layer_norm(hidden_states, weight=ln1_weight, bias=ln1_bias, epsilon=1e-12)

    # Multi-Head Self-Attention
    attn_out = ttnn_ast_attention(
        ln1_out, qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads, head_dim, device
    )
    ttnn.deallocate(ln1_out)

    # Residual connection 1
    hidden_states = ttnn.add(hidden_states, attn_out)
    ttnn.deallocate(attn_out)

    # Pre-LayerNorm 2
    ln2_out = ttnn.layer_norm(hidden_states, weight=ln2_weight, bias=ln2_bias, epsilon=1e-12)

    # Feed-Forward Network: Linear → GELU → Linear
    ff_out = ttnn.matmul(ln2_out, ff_weight1)
    ff_out = ttnn.add(ff_out, ff_bias1)
    ttnn.deallocate(ln2_out)

    ff_out = ttnn.gelu(ff_out)

    ff_out2 = ttnn.matmul(ff_out, ff_weight2)
    ttnn.deallocate(ff_out)
    ff_out2 = ttnn.add(ff_out2, ff_bias2)

    # Residual connection 2
    output = ttnn.add(hidden_states, ff_out2)
    ttnn.deallocate(hidden_states)
    ttnn.deallocate(ff_out2)

    return output


def ttnn_ast_model(
    patches: torch.Tensor,
    reference_model: ASTModel,
    device: ttnn.Device,
) -> torch.Tensor:
    """Run the full AST model through TTNN, returning logits on CPU.

    Executes embedding → 12 encoder blocks → final LayerNorm → mean-pool
    CLS and distillation tokens → classifier head.

    Args:
        patches: CPU tensor ``[batch, NUM_PATCHES, PATCH_DIM]``.
        reference_model: Pretrained PyTorch AST model (used for weight extraction).
        device: Target ttnn device.

    Returns:
        CPU tensor ``[batch, num_classes]`` — classification logits.
    """
    batch_size = patches.shape[0]
    state = reference_model.state_dict()
    config = reference_model.config
    head_dim = HIDDEN_SIZE // NUM_ATTENTION_HEADS

    # -----------------------------------------------------------------------
    # Embedding
    # -----------------------------------------------------------------------
    proj_weight = state["embeddings.patch_embeddings.projection.weight"]
    # Reshape conv weight [H, 1, 16, 16] → [256, H] for matmul
    proj_weight = proj_weight.view(HIDDEN_SIZE, PATCH_DIM).t()  # [256, 768]
    proj_bias = state["embeddings.patch_embeddings.projection.bias"].unsqueeze(0).unsqueeze(0)

    cls_token = state["embeddings.cls_token"]  # [1, 1, 768]
    dist_token = state["embeddings.distillation_token"]  # [1, 1, 768]
    pos_embed = state["embeddings.position_embeddings"]  # [1, 1214, 768]

    # Expand CLS and distillation tokens to batch size
    cls_token_expanded = cls_token.expand(batch_size, -1, -1)
    dist_token_expanded = dist_token.expand(batch_size, -1, -1)

    patches_tt = _to_ttnn_tile(patches, device)
    proj_weight_tt = _to_ttnn_tile(proj_weight, device)
    proj_bias_tt = _to_ttnn_tile(proj_bias, device)
    cls_tt = _to_ttnn_tile(cls_token_expanded, device)
    dist_tt = _to_ttnn_tile(dist_token_expanded, device)
    pos_tt = _to_ttnn_tile(pos_embed, device)

    hidden = ttnn_ast_patch_embedding(
        patches_tt, proj_weight_tt, proj_bias_tt, cls_tt, dist_tt, pos_tt, device
    )

    # Deallocate embedding weights
    for t in [patches_tt, proj_weight_tt, proj_bias_tt, cls_tt, dist_tt, pos_tt]:
        ttnn.deallocate(t)

    # -----------------------------------------------------------------------
    # Encoder blocks
    # -----------------------------------------------------------------------
    for layer_idx in range(NUM_ENCODER_LAYERS):
        prefix = f"encoder.layer.{layer_idx}"

        # LayerNorm 1
        ln1_w = state[f"{prefix}.layernorm_before.weight"].unsqueeze(0).unsqueeze(0)
        ln1_b = state[f"{prefix}.layernorm_before.bias"].unsqueeze(0).unsqueeze(0)

        # Attention — fuse Q, K, V into single weight/bias
        q_w = state[f"{prefix}.attention.attention.query.weight"]
        k_w = state[f"{prefix}.attention.attention.key.weight"]
        v_w = state[f"{prefix}.attention.attention.value.weight"]
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0).t()  # [H, 3H]

        q_b = state[f"{prefix}.attention.attention.query.bias"]
        k_b = state[f"{prefix}.attention.attention.key.bias"]
        v_b = state[f"{prefix}.attention.attention.value.bias"]
        qkv_b = torch.cat([q_b, k_b, v_b], dim=0).unsqueeze(0).unsqueeze(0)

        proj_w = state[f"{prefix}.attention.output.dense.weight"].t()
        proj_b = state[f"{prefix}.attention.output.dense.bias"].unsqueeze(0).unsqueeze(0)

        # LayerNorm 2
        ln2_w = state[f"{prefix}.layernorm_after.weight"].unsqueeze(0).unsqueeze(0)
        ln2_b = state[f"{prefix}.layernorm_after.bias"].unsqueeze(0).unsqueeze(0)

        # FFN
        ff_w1 = state[f"{prefix}.intermediate.dense.weight"].t()
        ff_b1 = state[f"{prefix}.intermediate.dense.bias"].unsqueeze(0).unsqueeze(0)
        ff_w2 = state[f"{prefix}.output.dense.weight"].t()
        ff_b2 = state[f"{prefix}.output.dense.bias"].unsqueeze(0).unsqueeze(0)

        # Move all to device
        ln1_w_tt = _to_ttnn_tile(ln1_w, device)
        ln1_b_tt = _to_ttnn_tile(ln1_b, device)
        qkv_w_tt = _to_ttnn_tile(qkv_w, device)
        qkv_b_tt = _to_ttnn_tile(qkv_b, device)
        proj_w_tt = _to_ttnn_tile(proj_w, device)
        proj_b_tt = _to_ttnn_tile(proj_b, device)
        ln2_w_tt = _to_ttnn_tile(ln2_w, device)
        ln2_b_tt = _to_ttnn_tile(ln2_b, device)
        ff_w1_tt = _to_ttnn_tile(ff_w1, device)
        ff_b1_tt = _to_ttnn_tile(ff_b1, device)
        ff_w2_tt = _to_ttnn_tile(ff_w2, device)
        ff_b2_tt = _to_ttnn_tile(ff_b2, device)

        hidden = ttnn_ast_encoder_block(
            hidden,
            ln1_w_tt, ln1_b_tt,
            qkv_w_tt, qkv_b_tt,
            proj_w_tt, proj_b_tt,
            ln2_w_tt, ln2_b_tt,
            ff_w1_tt, ff_b1_tt,
            ff_w2_tt, ff_b2_tt,
            NUM_ATTENTION_HEADS, head_dim, device,
        )

        # Deallocate layer weights
        for t in [
            ln1_w_tt, ln1_b_tt, qkv_w_tt, qkv_b_tt, proj_w_tt, proj_b_tt,
            ln2_w_tt, ln2_b_tt, ff_w1_tt, ff_b1_tt, ff_w2_tt, ff_b2_tt,
        ]:
            ttnn.deallocate(t)

    # -----------------------------------------------------------------------
    # Final LayerNorm
    # -----------------------------------------------------------------------
    final_ln_w = state["layernorm.weight"].unsqueeze(0).unsqueeze(0)
    final_ln_b = state["layernorm.bias"].unsqueeze(0).unsqueeze(0)
    final_ln_w_tt = _to_ttnn_tile(final_ln_w, device)
    final_ln_b_tt = _to_ttnn_tile(final_ln_b, device)

    hidden = ttnn.layer_norm(hidden, weight=final_ln_w_tt, bias=final_ln_b_tt, epsilon=1e-12)
    ttnn.deallocate(final_ln_w_tt)
    ttnn.deallocate(final_ln_b_tt)

    # -----------------------------------------------------------------------
    # Pooling: mean of CLS and distillation tokens → [B, H]
    # -----------------------------------------------------------------------
    hidden_cpu = _from_ttnn(hidden, (batch_size, NUM_TOKENS, HIDDEN_SIZE))
    ttnn.deallocate(hidden)

    cls_out = hidden_cpu[:, 0, :]  # [B, H]
    dist_out = hidden_cpu[:, 1, :]  # [B, H]
    pooled = (cls_out + dist_out) / 2.0  # [B, H]

    # -----------------------------------------------------------------------
    # Classifier head (small linear — run on CPU for simplicity)
    # -----------------------------------------------------------------------
    classifier_weight = state.get("classifier.dense.weight", state.get("classifier.weight", None))
    classifier_bias = state.get("classifier.dense.bias", state.get("classifier.bias", None))

    if classifier_weight is not None:
        logits = torch.nn.functional.linear(pooled, classifier_weight, classifier_bias)
    else:
        logits = pooled

    return logits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_model():
    """Load the pretrained AST model once per test module."""
    return _load_reference_model()


@pytest.fixture(scope="module")
def reference_config():
    """Load the AST configuration once per test module."""
    return ASTConfig.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1])
def test_ast_pcc(device, batch_size, reference_model):
    """Full model PCC validation against PyTorch reference.

    1. Create dummy spectrogram input [batch, 1024, 128].
    2. Run PyTorch reference → golden logits.
    3. Unfold patches on CPU → [batch, 1212, 256].
    4. Convert to ttnn and run TTNN model.
    5. Assert PCC >= 0.99.
    """
    spectrogram = _create_dummy_spectrogram(batch_size)

    # Golden reference
    with torch.no_grad():
        golden_output = reference_model(spectrogram)
        golden_hidden = golden_output.last_hidden_state  # [B, 1214, 768]
        cls_out = golden_hidden[:, 0, :]
        dist_out = golden_hidden[:, 1, :]
        golden_pooled = (cls_out + dist_out) / 2.0

    # TTNN path
    patches = _unfold_spectrogram(spectrogram)
    ttnn_logits = ttnn_ast_model(patches, reference_model, device)

    # Compare pooled outputs (before classifier, since classifier may differ)
    pcc = compute_pcc(golden_pooled, ttnn_logits[:, :HIDDEN_SIZE] if ttnn_logits.shape[-1] > HIDDEN_SIZE else ttnn_logits)

    print(f"\n[test_ast_pcc] PCC = {pcc:.6f} (threshold = {EXPECTED_PCC})")
    assert pcc >= EXPECTED_PCC, (
        f"Full model PCC {pcc:.6f} is below threshold {EXPECTED_PCC}"
    )


@pytest.mark.parametrize("batch_size", [1])
def test_ast_top1_accuracy(device, batch_size, reference_model):
    """Verify top-1 prediction matches PyTorch reference.

    Ensures that TTNN and PyTorch produce the same argmax class prediction,
    validating functional correctness end-to-end.
    """
    spectrogram = _create_dummy_spectrogram(batch_size)

    # Golden reference
    with torch.no_grad():
        golden_output = reference_model(spectrogram)
        golden_hidden = golden_output.last_hidden_state
        golden_pooled = (golden_hidden[:, 0, :] + golden_hidden[:, 1, :]) / 2.0

    # TTNN path
    patches = _unfold_spectrogram(spectrogram)
    ttnn_logits = ttnn_ast_model(patches, reference_model, device)

    # Get top-1 predictions
    state = reference_model.state_dict()
    classifier_weight = state.get("classifier.dense.weight", state.get("classifier.weight", None))
    classifier_bias = state.get("classifier.dense.bias", state.get("classifier.bias", None))

    if classifier_weight is not None:
        golden_logits = torch.nn.functional.linear(golden_pooled, classifier_weight, classifier_bias)
    else:
        golden_logits = golden_pooled

    golden_top1 = golden_logits.argmax(dim=-1)
    ttnn_top1 = ttnn_logits.argmax(dim=-1)

    print(f"\n[test_ast_top1_accuracy] Golden top-1: {golden_top1.tolist()}, TTNN top-1: {ttnn_top1.tolist()}")
    assert torch.equal(golden_top1, ttnn_top1), (
        f"Top-1 mismatch: golden={golden_top1.tolist()}, ttnn={ttnn_top1.tolist()}"
    )


@pytest.mark.parametrize("batch_size", [1])
def test_ast_embedding_pcc(device, batch_size, reference_model):
    """Test patch embedding + positional embedding PCC.

    Validates that the TTNN embedding layer (linear projection, CLS/distillation
    token prepend, positional embedding addition) matches the PyTorch reference.
    """
    spectrogram = _create_dummy_spectrogram(batch_size)

    # Golden embedding output
    golden_embeddings = _get_reference_embeddings(reference_model, spectrogram)

    # TTNN embedding path
    state = reference_model.state_dict()
    patches = _unfold_spectrogram(spectrogram)

    proj_weight = state["embeddings.patch_embeddings.projection.weight"]
    proj_weight = proj_weight.view(HIDDEN_SIZE, PATCH_DIM).t()
    proj_bias = state["embeddings.patch_embeddings.projection.bias"].unsqueeze(0).unsqueeze(0)
    cls_token = state["embeddings.cls_token"].expand(batch_size, -1, -1)
    dist_token = state["embeddings.distillation_token"].expand(batch_size, -1, -1)
    pos_embed = state["embeddings.position_embeddings"]

    patches_tt = _to_ttnn_tile(patches, device)
    proj_w_tt = _to_ttnn_tile(proj_weight, device)
    proj_b_tt = _to_ttnn_tile(proj_bias, device)
    cls_tt = _to_ttnn_tile(cls_token, device)
    dist_tt = _to_ttnn_tile(dist_token, device)
    pos_tt = _to_ttnn_tile(pos_embed, device)

    ttnn_embeddings = ttnn_ast_patch_embedding(
        patches_tt, proj_w_tt, proj_b_tt, cls_tt, dist_tt, pos_tt, device
    )

    ttnn_embed_cpu = _from_ttnn(ttnn_embeddings, (batch_size, NUM_TOKENS, HIDDEN_SIZE))

    # Deallocate
    for t in [patches_tt, proj_w_tt, proj_b_tt, cls_tt, dist_tt, pos_tt, ttnn_embeddings]:
        ttnn.deallocate(t)

    pcc = compute_pcc(golden_embeddings, ttnn_embed_cpu)
    print(f"\n[test_ast_embedding_pcc] PCC = {pcc:.6f} (threshold = {EXPECTED_PCC})")
    assert pcc >= EXPECTED_PCC, (
        f"Embedding PCC {pcc:.6f} is below threshold {EXPECTED_PCC}"
    )


@pytest.mark.parametrize("batch_size", [1])
def test_ast_single_encoder_block_pcc(device, batch_size, reference_model):
    """Test single encoder layer PCC.

    Runs layer 0 of the AST encoder in both PyTorch and TTNN, comparing
    the output hidden states. Uses the golden embedding output as input
    to isolate encoder block accuracy.
    """
    spectrogram = _create_dummy_spectrogram(batch_size)

    # Get golden embedding output as input to encoder block
    golden_embeddings = _get_reference_embeddings(reference_model, spectrogram)

    # Golden encoder block 0 output
    golden_block_out = _get_reference_encoder_block_output(reference_model, golden_embeddings, layer_idx=0)

    # TTNN encoder block path
    state = reference_model.state_dict()
    head_dim = HIDDEN_SIZE // NUM_ATTENTION_HEADS
    prefix = "encoder.layer.0"

    ln1_w = state[f"{prefix}.layernorm_before.weight"].unsqueeze(0).unsqueeze(0)
    ln1_b = state[f"{prefix}.layernorm_before.bias"].unsqueeze(0).unsqueeze(0)

    q_w = state[f"{prefix}.attention.attention.query.weight"]
    k_w = state[f"{prefix}.attention.attention.key.weight"]
    v_w = state[f"{prefix}.attention.attention.value.weight"]
    qkv_w = torch.cat([q_w, k_w, v_w], dim=0).t()

    q_b = state[f"{prefix}.attention.attention.query.bias"]
    k_b = state[f"{prefix}.attention.attention.key.bias"]
    v_b = state[f"{prefix}.attention.attention.value.bias"]
    qkv_b = torch.cat([q_b, k_b, v_b], dim=0).unsqueeze(0).unsqueeze(0)

    proj_w = state[f"{prefix}.attention.output.dense.weight"].t()
    proj_b = state[f"{prefix}.attention.output.dense.bias"].unsqueeze(0).unsqueeze(0)

    ln2_w = state[f"{prefix}.layernorm_after.weight"].unsqueeze(0).unsqueeze(0)
    ln2_b = state[f"{prefix}.layernorm_after.bias"].unsqueeze(0).unsqueeze(0)

    ff_w1 = state[f"{prefix}.intermediate.dense.weight"].t()
    ff_b1 = state[f"{prefix}.intermediate.dense.bias"].unsqueeze(0).unsqueeze(0)
    ff_w2 = state[f"{prefix}.output.dense.weight"].t()
    ff_b2 = state[f"{prefix}.output.dense.bias"].unsqueeze(0).unsqueeze(0)

    # Move input and weights to device
    hidden_tt = _to_ttnn_tile(golden_embeddings, device)
    ln1_w_tt = _to_ttnn_tile(ln1_w, device)
    ln1_b_tt = _to_ttnn_tile(ln1_b, device)
    qkv_w_tt = _to_ttnn_tile(qkv_w, device)
    qkv_b_tt = _to_ttnn_tile(qkv_b, device)
    proj_w_tt = _to_ttnn_tile(proj_w, device)
    proj_b_tt = _to_ttnn_tile(proj_b, device)
    ln2_w_tt = _to_ttnn_tile(ln2_w, device)
    ln2_b_tt = _to_ttnn_tile(ln2_b, device)
    ff_w1_tt = _to_ttnn_tile(ff_w1, device)
    ff_b1_tt = _to_ttnn_tile(ff_b1, device)
    ff_w2_tt = _to_ttnn_tile(ff_w2, device)
    ff_b2_tt = _to_ttnn_tile(ff_b2, device)

    ttnn_block_out = ttnn_ast_encoder_block(
        hidden_tt,
        ln1_w_tt, ln1_b_tt,
        qkv_w_tt, qkv_b_tt,
        proj_w_tt, proj_b_tt,
        ln2_w_tt, ln2_b_tt,
        ff_w1_tt, ff_b1_tt,
        ff_w2_tt, ff_b2_tt,
        NUM_ATTENTION_HEADS, head_dim, device,
    )

    ttnn_block_cpu = _from_ttnn(ttnn_block_out, (batch_size, NUM_TOKENS, HIDDEN_SIZE))

    # Deallocate
    for t in [
        ln1_w_tt, ln1_b_tt, qkv_w_tt, qkv_b_tt, proj_w_tt, proj_b_tt,
        ln2_w_tt, ln2_b_tt, ff_w1_tt, ff_b1_tt, ff_w2_tt, ff_b2_tt,
        ttnn_block_out,
    ]:
        ttnn.deallocate(t)

    pcc = compute_pcc(golden_block_out, ttnn_block_cpu)
    print(f"\n[test_ast_single_encoder_block_pcc] PCC = {pcc:.6f} (threshold = {EXPECTED_PCC})")
    assert pcc >= EXPECTED_PCC, (
        f"Encoder block 0 PCC {pcc:.6f} is below threshold {EXPECTED_PCC}"
    )
