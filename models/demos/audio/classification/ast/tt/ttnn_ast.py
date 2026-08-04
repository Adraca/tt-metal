# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Audio Spectrogram Transformer (AST) implementation.

Bounty: tenstorrent/tt-metal#52054
Reference model: MIT/ast-finetuned-audioset-10-10-0.4593

Architecture:
    - Pre-LayerNorm transformer encoder (12 layers)
    - Hidden: 768, Heads: 12, Head dim: 64, Intermediate: 3072
    - Input: [1, 1024, 128] mel spectrogram
    - Patch: 16×16 stride 10×10 → 1212 patches + CLS + DIST = 1214 tokens
    - Tile-aligned sequence length: 1216 (pad 1214 → nearest multiple of 32)
    - Classification: avg(CLS, DIST) → LayerNorm → Linear(768, 527)
"""

import math
from typing import Optional

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 768
NUM_HEADS = 12
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 64
INTERMEDIATE_SIZE = 3072
NUM_LAYERS = 12
PATCH_SIZE = 16
PATCH_STRIDE = 10
NUM_FREQ_PATCHES = 12   # (128 - 16) // 10 + 1
NUM_TIME_PATCHES = 101  # (1024 - 16) // 10 + 1
NUM_PATCHES = NUM_FREQ_PATCHES * NUM_TIME_PATCHES  # 1212
TRUE_SEQ_LEN = NUM_PATCHES + 2  # 1214 (CLS + DIST)
TILE_SEQ_LEN = 1216  # nearest multiple of 32 >= 1214
NUM_CLASSES = 527
TILE_NUM_CLASSES = 544  # nearest multiple of 32 >= 527
PATCH_DIM = PATCH_SIZE * PATCH_SIZE  # 256
LN_EPS = 1e-12
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)


# ---------------------------------------------------------------------------
# custom_preprocessor — fuse QKV, transpose weights, flatten conv kernel
# ---------------------------------------------------------------------------
def custom_preprocessor(model, name="", custom_preprocessor=None, **kwargs):
    """Preprocess HuggingFace AST state dict for TTNN consumption.

    Transformations applied:
        1. Fuse per-layer Q, K, V weights into a single QKV weight [768, 2304]
           and fuse Q, K, V biases into a single QKV bias [2304].
        2. Transpose all dense/linear weights for TTNN matmul convention
           (TTNN expects weight shape [in_features, out_features]).
        3. Flatten Conv2d projection weight [768, 1, 16, 16] → [256, 768]
           for use as a linear layer after CPU-side unfold.

    Args:
        model: HuggingFace ASTForAudioClassification model instance.
        name: Parameter namespace prefix (unused, kept for API compat).

    Returns:
        dict: Modified parameters dictionary ready for TTNN model construction.
    """
    parameters = {}
    state_dict = model.state_dict()

    prefix = "audio_spectrogram_transformer."

    # ----- Embeddings -----
    emb_prefix = prefix + "embeddings."
    parameters["embeddings"] = {}
    parameters["embeddings"]["cls_token"] = state_dict[emb_prefix + "cls_token"]
    parameters["embeddings"]["distillation_token"] = state_dict[emb_prefix + "distillation_token"]
    parameters["embeddings"]["position_embeddings"] = state_dict[emb_prefix + "position_embeddings"]

    # Flatten Conv2d projection: [768, 1, 16, 16] → [256, 768]
    conv_weight = state_dict[emb_prefix + "patch_embeddings.projection.weight"]
    # conv_weight shape: [out_channels=768, in_channels=1, kH=16, kW=16]
    # Flatten to [768, 256] then transpose to [256, 768] for TTNN linear
    conv_weight_flat = conv_weight.reshape(HIDDEN_SIZE, PATCH_DIM).t().contiguous()
    parameters["embeddings"]["projection_weight"] = conv_weight_flat
    parameters["embeddings"]["projection_bias"] = state_dict[emb_prefix + "patch_embeddings.projection.bias"]

    # ----- Encoder layers -----
    parameters["encoder"] = {}
    for i in range(NUM_LAYERS):
        layer_prefix = f"{prefix}encoder.layer.{i}."
        layer_params = {}

        # -- Fuse QKV --
        q_weight = state_dict[layer_prefix + "attention.attention.query.weight"]  # [768, 768]
        k_weight = state_dict[layer_prefix + "attention.attention.key.weight"]
        v_weight = state_dict[layer_prefix + "attention.attention.value.weight"]
        q_bias = state_dict[layer_prefix + "attention.attention.query.bias"]  # [768]
        k_bias = state_dict[layer_prefix + "attention.attention.key.bias"]
        v_bias = state_dict[layer_prefix + "attention.attention.value.bias"]

        # Fused QKV weight: cat along output dim → [3*768, 768], then transpose → [768, 2304]
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # [2304, 768]
        qkv_weight = qkv_weight.t().contiguous()  # [768, 2304]
        layer_params["qkv_weight"] = qkv_weight

        # Fused QKV bias: [2304]
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)  # [2304]
        layer_params["qkv_bias"] = qkv_bias

        # -- Attention output dense --
        attn_out_weight = state_dict[layer_prefix + "attention.output.dense.weight"]  # [768, 768]
        layer_params["attn_output_weight"] = attn_out_weight.t().contiguous()  # [768, 768]
        layer_params["attn_output_bias"] = state_dict[layer_prefix + "attention.output.dense.bias"]

        # -- LayerNorms --
        layer_params["layernorm_before_weight"] = state_dict[layer_prefix + "layernorm_before.weight"]
        layer_params["layernorm_before_bias"] = state_dict[layer_prefix + "layernorm_before.bias"]
        layer_params["layernorm_after_weight"] = state_dict[layer_prefix + "layernorm_after.weight"]
        layer_params["layernorm_after_bias"] = state_dict[layer_prefix + "layernorm_after.bias"]

        # -- MLP intermediate dense --
        inter_weight = state_dict[layer_prefix + "intermediate.dense.weight"]  # [3072, 768]
        layer_params["intermediate_weight"] = inter_weight.t().contiguous()  # [768, 3072]
        layer_params["intermediate_bias"] = state_dict[layer_prefix + "intermediate.dense.bias"]

        # -- MLP output dense --
        out_weight = state_dict[layer_prefix + "output.dense.weight"]  # [768, 3072]
        layer_params["output_weight"] = out_weight.t().contiguous()  # [3072, 768]
        layer_params["output_bias"] = state_dict[layer_prefix + "output.dense.bias"]

        parameters["encoder"][str(i)] = layer_params

    # ----- Final encoder LayerNorm -----
    parameters["layernorm_weight"] = state_dict[prefix + "layernorm.weight"]
    parameters["layernorm_bias"] = state_dict[prefix + "layernorm.bias"]

    # ----- Classifier head -----
    parameters["classifier"] = {}
    parameters["classifier"]["layernorm_weight"] = state_dict["classifier.layernorm.weight"]
    parameters["classifier"]["layernorm_bias"] = state_dict["classifier.layernorm.bias"]

    # Classifier dense: [527, 768] → transpose to [768, 527], then pad to [768, 544]
    cls_weight = state_dict["classifier.dense.weight"]  # [527, 768]
    cls_weight = cls_weight.t().contiguous()  # [768, 527]
    # Pad output dim to tile-aligned 544
    pad_cols = TILE_NUM_CLASSES - NUM_CLASSES  # 17
    cls_weight_padded = torch.nn.functional.pad(cls_weight, (0, pad_cols), value=0.0)  # [768, 544]
    parameters["classifier"]["dense_weight"] = cls_weight_padded

    cls_bias = state_dict["classifier.dense.bias"]  # [527]
    cls_bias_padded = torch.nn.functional.pad(cls_bias, (0, pad_cols), value=0.0)  # [544]
    parameters["classifier"]["dense_bias"] = cls_bias_padded

    return parameters


def _to_device(tensor_or_param, device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    """Move a PyTorch tensor or TTNN tensor to device with specified layout and dtype."""
    if isinstance(tensor_or_param, torch.Tensor):
        t = ttnn.from_torch(tensor_or_param, dtype=dtype, layout=layout)
        return ttnn.to_device(t, device)
    return tensor_or_param


def _prepare_parameters_on_device(parameters, device):
    """Recursively move all parameter tensors in a nested dict to device.

    Args:
        parameters: Nested dict of torch.Tensors from custom_preprocessor.
        device: TTNN device handle.

    Returns:
        Nested dict with all tensors on device in TILE_LAYOUT bfloat16.
    """
    on_device = {}
    for key, value in parameters.items():
        if isinstance(value, dict):
            on_device[key] = _prepare_parameters_on_device(value, device)
        elif isinstance(value, torch.Tensor):
            on_device[key] = _to_device(value, device)
        else:
            on_device[key] = value
    return on_device


# ---------------------------------------------------------------------------
# TtASTEmbeddings
# ---------------------------------------------------------------------------
class TtASTEmbeddings:
    """TTNN AST patch embedding + positional embedding layer.

    Performs:
        1. Linear projection of unfolded patches  [B, 1212, 256] → [B, 1, 1212, 768]
        2. Prepend CLS and distillation tokens     → [B, 1, 1214, 768]
        3. Add positional embeddings
        4. Pad sequence 1214 → 1216 for tile alignment

    Args:
        device: TTNN device handle.
        parameters: Dict containing embedding parameters on device.
        hidden_size: Transformer hidden dimension (default: 768).
    """

    def __init__(self, device, parameters, hidden_size: int = HIDDEN_SIZE):
        self.device = device
        self.hidden_size = hidden_size

        self.cls_token = parameters["cls_token"]                       # [1, 1, 768]
        self.distillation_token = parameters["distillation_token"]     # [1, 1, 768]
        self.position_embeddings = parameters["position_embeddings"]   # [1, 1214, 768]
        self.projection_weight = parameters["projection_weight"]       # [256, 768]
        self.projection_bias = parameters["projection_bias"]           # [768]

    def __call__(self, input_values):
        """Embed unfolded spectrogram patches.

        Args:
            input_values: ttnn tensor of unfolded patches, shape [B, 1, 1212, 256]
                on device, TILE_LAYOUT, bfloat16. Patches are already extracted
                on CPU via torch.nn.functional.unfold.

        Returns:
            ttnn tensor of shape [B, 1, 1216, 768] on device (tile-aligned).
        """
        batch_size = input_values.shape[0]

        # --- Project patches: [B, 1, 1212, 256] @ [256, 768] → [B, 1, 1212, 768] ---
        patch_embeddings = ttnn.linear(
            input_values,
            self.projection_weight,
            bias=self.projection_bias,
        )
        ttnn.deallocate(input_values)

        # --- Prepend CLS and distillation tokens ---
        # Expand special tokens to batch: [1, 1, 768] → [B, 1, 768] via repeat
        # We need shape [B, 1, 1, 768] for concat along seq dim (dim=2)
        cls_tokens = ttnn.repeat(self.cls_token, ttnn.Shape([batch_size, 1, 1]))
        dist_tokens = ttnn.repeat(self.distillation_token, ttnn.Shape([batch_size, 1, 1]))

        # Reshape special tokens to 4D: [B, 1, 768] → [B, 1, 1, 768]
        cls_tokens = ttnn.reshape(cls_tokens, (batch_size, 1, 1, self.hidden_size))
        dist_tokens = ttnn.reshape(dist_tokens, (batch_size, 1, 1, self.hidden_size))

        # Concat: [B,1,1,768] + [B,1,1,768] + [B,1,1212,768] → [B,1,1214,768]
        embeddings = ttnn.concat([cls_tokens, dist_tokens, patch_embeddings], dim=2)
        ttnn.deallocate(cls_tokens)
        ttnn.deallocate(dist_tokens)
        ttnn.deallocate(patch_embeddings)

        # --- Add position embeddings ---
        # position_embeddings: [1, 1214, 768] → reshape to [1, 1, 1214, 768]
        pos_emb = ttnn.reshape(self.position_embeddings, (1, 1, TRUE_SEQ_LEN, self.hidden_size))
        embeddings = ttnn.add(embeddings, pos_emb)

        # --- Pad sequence 1214 → 1216 for tile alignment ---
        embeddings = ttnn.pad(embeddings, padding=((0, 0), (0, 0), (0, TILE_SEQ_LEN - TRUE_SEQ_LEN), (0, 0)), value=0.0)

        return embeddings


# ---------------------------------------------------------------------------
# TtASTSelfAttention
# ---------------------------------------------------------------------------
class TtASTSelfAttention:
    """TTNN fused multi-head self-attention for AST.

    Uses a single fused QKV projection, splits heads via
    ttnn.transformer.split_query_key_value_and_split_heads,
    performs scaled dot-product attention manually, then
    concatenates heads and projects output.

    Args:
        device: TTNN device handle.
        parameters: Dict containing layer attention parameters on device.
    """

    def __init__(self, device, parameters):
        self.device = device
        self.qkv_weight = parameters["qkv_weight"]          # [768, 2304]
        self.qkv_bias = parameters["qkv_bias"]              # [2304]
        self.attn_output_weight = parameters["attn_output_weight"]  # [768, 768]
        self.attn_output_bias = parameters["attn_output_bias"]      # [768]

    def __call__(self, hidden_states):
        """Forward pass of multi-head self-attention.

        Args:
            hidden_states: [B, 1, S, 768] on device, where S = TILE_SEQ_LEN (1216).

        Returns:
            ttnn tensor [B, 1, S, 768] — attention output.
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[2]

        # --- Fused QKV projection ---
        # [B, 1, S, 768] @ [768, 2304] → [B, 1, S, 2304]
        qkv = ttnn.linear(hidden_states, self.qkv_weight, bias=self.qkv_bias)

        # --- Split into Q, K, V and split heads ---
        # Returns: Q [B, num_heads, S, head_dim], K [B, num_heads, S, head_dim], V same
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            num_heads=NUM_HEADS,
        )
        ttnn.deallocate(qkv)

        # --- Scaled dot-product attention ---
        # attn_weights = Q @ K^T * scale  →  [B, num_heads, S, S]
        key_transposed = ttnn.transpose(key, -2, -1)
        attn_weights = ttnn.matmul(query, key_transposed)
        ttnn.deallocate(key_transposed)
        ttnn.deallocate(query)

        attn_weights = ttnn.multiply(attn_weights, ATTN_SCALE)

        # Softmax along last dim
        attn_probs = ttnn.softmax(attn_weights, dim=-1)
        ttnn.deallocate(attn_weights)

        # attn_output = attn_probs @ V  →  [B, num_heads, S, head_dim]
        attn_output = ttnn.matmul(attn_probs, value)
        ttnn.deallocate(attn_probs)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        # --- Concatenate heads ---
        # [B, num_heads, S, head_dim] → [B, 1, S, 768]
        attn_output = ttnn.transformer.concatenate_heads(attn_output)

        # --- Output projection ---
        # [B, 1, S, 768] @ [768, 768] → [B, 1, S, 768]
        attn_output = ttnn.linear(attn_output, self.attn_output_weight, bias=self.attn_output_bias)

        return attn_output


# ---------------------------------------------------------------------------
# TtASTMLP
# ---------------------------------------------------------------------------
class TtASTMLP:
    """TTNN MLP block for AST encoder layer.

    Linear(768→3072) → GELU → Linear(3072→768)

    Args:
        device: TTNN device handle.
        parameters: Dict containing MLP parameters on device.
    """

    def __init__(self, device, parameters):
        self.device = device
        self.intermediate_weight = parameters["intermediate_weight"]  # [768, 3072]
        self.intermediate_bias = parameters["intermediate_bias"]      # [3072]
        self.output_weight = parameters["output_weight"]              # [3072, 768]
        self.output_bias = parameters["output_bias"]                  # [768]

    def __call__(self, hidden_states):
        """Forward pass of two-layer MLP with GELU activation.

        Args:
            hidden_states: [B, 1, S, 768] on device.

        Returns:
            ttnn tensor [B, 1, S, 768].
        """
        # Intermediate: [B, 1, S, 768] → [B, 1, S, 3072]
        hidden = ttnn.linear(hidden_states, self.intermediate_weight, bias=self.intermediate_bias)

        # GELU activation
        hidden = ttnn.gelu(hidden)

        # Output: [B, 1, S, 3072] → [B, 1, S, 768]
        output = ttnn.linear(hidden, self.output_weight, bias=self.output_bias)
        ttnn.deallocate(hidden)

        return output


# ---------------------------------------------------------------------------
# TtASTEncoderLayer
# ---------------------------------------------------------------------------
class TtASTEncoderLayer:
    """Single pre-LayerNorm transformer encoder layer for AST.

    Architecture (pre-LN):
        x_normed = LayerNorm_before(x)
        x = x + SelfAttention(x_normed)
        x_normed = LayerNorm_after(x)
        x = x + MLP(x_normed)

    Args:
        device: TTNN device handle.
        parameters: Dict containing all layer parameters on device.
    """

    def __init__(self, device, parameters):
        self.device = device
        self.attention = TtASTSelfAttention(device, parameters)
        self.mlp = TtASTMLP(device, parameters)

        self.ln_before_weight = parameters["layernorm_before_weight"]  # [768]
        self.ln_before_bias = parameters["layernorm_before_bias"]
        self.ln_after_weight = parameters["layernorm_after_weight"]    # [768]
        self.ln_after_bias = parameters["layernorm_after_bias"]

    def __call__(self, hidden_states):
        """Forward pass of a single encoder layer.

        Args:
            hidden_states: [B, 1, S, 768] on device.

        Returns:
            ttnn tensor [B, 1, S, 768] — layer output.
        """
        # --- Pre-LN attention block ---
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.ln_before_weight,
            bias=self.ln_before_bias,
            epsilon=LN_EPS,
        )

        attn_output = self.attention(hidden_states)
        ttnn.deallocate(hidden_states)

        hidden_states = ttnn.add(residual, attn_output)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_output)

        # --- Pre-LN MLP block ---
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.ln_after_weight,
            bias=self.ln_after_bias,
            epsilon=LN_EPS,
        )

        mlp_output = self.mlp(hidden_states)
        ttnn.deallocate(hidden_states)

        hidden_states = ttnn.add(residual, mlp_output)
        ttnn.deallocate(residual)
        ttnn.deallocate(mlp_output)

        return hidden_states


# ---------------------------------------------------------------------------
# TtASTModel
# ---------------------------------------------------------------------------
class TtASTModel:
    """TTNN AST backbone: embeddings → 12× encoder layers → final LayerNorm.

    Args:
        device: TTNN device handle.
        parameters: Dict containing all model parameters on device.
    """

    def __init__(self, device, parameters):
        self.device = device
        self.embeddings = TtASTEmbeddings(device, parameters["embeddings"])

        self.encoder_layers = []
        for i in range(NUM_LAYERS):
            layer = TtASTEncoderLayer(device, parameters["encoder"][str(i)])
            self.encoder_layers.append(layer)

        self.ln_weight = parameters["layernorm_weight"]  # [768]
        self.ln_bias = parameters["layernorm_bias"]

    def __call__(self, input_values):
        """Forward pass of the AST backbone.

        Args:
            input_values: ttnn tensor of unfolded patches [B, 1, 1212, 256] on device.

        Returns:
            ttnn tensor [B, 1, TILE_SEQ_LEN, 768] — encoded sequence.
        """
        hidden_states = self.embeddings(input_values)

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)

        # Final LayerNorm
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=self.ln_weight,
            bias=self.ln_bias,
            epsilon=LN_EPS,
        )

        return hidden_states


# ---------------------------------------------------------------------------
# TtASTForAudioClassification
# ---------------------------------------------------------------------------
class TtASTForAudioClassification:
    """TTNN AST for audio classification (527 AudioSet classes).

    Classification head:
        1. Extract CLS token (index 0) and distillation token (index 1)
        2. Average: (CLS + DIST) * 0.5
        3. LayerNorm → Linear(768, 527)
        4. Slice logits from tile-padded 544 back to 527

    Args:
        device: TTNN device handle.
        parameters: Dict containing all parameters on device.
    """

    def __init__(self, device, parameters):
        self.device = device
        self.ast_model = TtASTModel(device, parameters)

        self.cls_ln_weight = parameters["classifier"]["layernorm_weight"]  # [768]
        self.cls_ln_bias = parameters["classifier"]["layernorm_bias"]
        self.cls_dense_weight = parameters["classifier"]["dense_weight"]   # [768, 544]
        self.cls_dense_bias = parameters["classifier"]["dense_bias"]       # [544]

    def __call__(self, input_values):
        """Forward pass: backbone → classification head.

        Args:
            input_values: ttnn tensor of unfolded patches [B, 1, 1212, 256] on device.

        Returns:
            ttnn tensor [B, 1, 1, 527] — classification logits (unpadded).
        """
        batch_size = input_values.shape[0]

        # --- Backbone ---
        # Output: [B, 1, TILE_SEQ_LEN, 768]
        hidden_states = self.ast_model(input_values)

        # --- Extract CLS token (seq index 0) and DIST token (seq index 1) ---
        # ttnn.slice: [B, 1, TILE_SEQ_LEN, 768] → [B, 1, 1, 768]
        cls_token = ttnn.slice(
            hidden_states,
            begins=(0, 0, 0, 0),
            ends=(batch_size, 1, 1, HIDDEN_SIZE),
        )
        dist_token = ttnn.slice(
            hidden_states,
            begins=(0, 0, 1, 0),
            ends=(batch_size, 1, 2, HIDDEN_SIZE),
        )
        ttnn.deallocate(hidden_states)

        # --- Average CLS and DIST tokens ---
        pooled = ttnn.add(cls_token, dist_token)
        ttnn.deallocate(cls_token)
        ttnn.deallocate(dist_token)
        pooled = ttnn.multiply(pooled, 0.5)

        # --- Classifier head ---
        # LayerNorm
        pooled = ttnn.layer_norm(
            pooled,
            weight=self.cls_ln_weight,
            bias=self.cls_ln_bias,
            epsilon=LN_EPS,
        )

        # Linear: [B, 1, 1, 768] @ [768, 544] → [B, 1, 1, 544]
        logits = ttnn.linear(pooled, self.cls_dense_weight, bias=self.cls_dense_bias)
        ttnn.deallocate(pooled)

        # --- Unpad num_classes: 544 → 527 ---
        logits = ttnn.slice(
            logits,
            begins=(0, 0, 0, 0),
            ends=(batch_size, 1, 1, NUM_CLASSES),
        )

        return logits


# ---------------------------------------------------------------------------
# Spectrogram preprocessing (CPU-side unfold)
# ---------------------------------------------------------------------------
def unfold_spectrogram(input_values: torch.Tensor) -> torch.Tensor:
    """Unfold a mel spectrogram into patches on CPU.

    Applies torch.nn.functional.unfold with kernel_size=16, stride=10
    to extract 1212 patches of dimension 256 from the spectrogram.

    Args:
        input_values: torch.Tensor of shape [B, 1, 1024, 128] — mel spectrogram.

    Returns:
        torch.Tensor of shape [B, 1, 1212, 256] — unfolded patches,
        ready for conversion to TTNN tensor.
    """
    batch_size = input_values.shape[0]

    # unfold expects [B, C, H, W] with C=1
    # kernel_size = (16, 16), stride = (10, 10)
    patches = torch.nn.functional.unfold(
        input_values,
        kernel_size=(PATCH_SIZE, PATCH_SIZE),
        stride=(PATCH_STRIDE, PATCH_STRIDE),
    )
    # patches shape: [B, 256, 1212]

    # Transpose to [B, 1212, 256]
    patches = patches.transpose(1, 2).contiguous()

    # Reshape to 4D for TTNN: [B, 1, 1212, 256]
    patches = patches.unsqueeze(1)

    return patches


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def create_ast_model(device, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
    """Load a HuggingFace AST model, preprocess parameters, and create TTNN model.

    Args:
        device: TTNN device handle.
        model_name: HuggingFace model identifier.

    Returns:
        Tuple of (TtASTForAudioClassification, HF_model):
            - TTNN model ready for inference
            - Reference HF model for validation
    """
    from transformers import ASTForAudioClassification

    hf_model = ASTForAudioClassification.from_pretrained(model_name)
    hf_model.eval()

    # Preprocess: fuse QKV, transpose weights, flatten conv
    parameters = custom_preprocessor(hf_model)

    # Move all parameters to device
    device_parameters = _prepare_parameters_on_device(parameters, device)

    # Build TTNN model
    tt_model = TtASTForAudioClassification(device, device_parameters)

    return tt_model, hf_model
