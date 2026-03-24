# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    comp_pcc,
    fa_rand,
    get_chunk_size,
    nearest_n,
    nearest_pow_2,
    num_to_corerange,
)


def test_mask_dtype_bf16_then_bfp8(device):
    # Program cache
    device.enable_program_cache()
    device.clear_program_cache()

    # Shape + grid
    b, nh, nkv, s, d = 2, 8, 1, 1024, 64
    grid_size = (8, 4)
    grid = device.compute_with_storage_grid_size()
    if grid_size[0] > grid.x or grid_size[1] > grid.y:
        pytest.skip(f"Need compute grid at least {grid_size}, got {grid}")

    torch.manual_seed(20250322)

    # SDPA configs
    padded_heads = nearest_pow_2(nearest_n(nh, n=32))
    dram = ttnn.DRAM_MEMORY_CONFIG
    q_dtype = ttnn.bfloat16
    kv_dtype = ttnn.bfloat8_b
    k_chunk_size = get_chunk_size(s // 4 + 1, s)
    scale = d**-0.5
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Inputs
    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)
    tt_k = ttnn.as_tensor(k, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.as_tensor(v, device=device, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    mask_torch = torch.bernoulli(torch.full((b, nh, 1, s), 0.25)) * torch.finfo(torch.float32).min
    q = fa_rand(1, b, nh, d)
    tt_q = ttnn.as_tensor(
        q[:, :, :nh],
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )

    # PyTorch golden
    q_s = q[:, :, :nh, :].permute(1, 2, 0, 3)
    k_s = k[:, :, :s, :]
    k_s = torch.cat([k_s[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    v_s = v[:, :, :s, :]
    v_s = torch.cat([v_s[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    m_s = mask_torch[:, :nh, :, :]
    ref = (
        torch.nn.functional.scaled_dot_product_attention(q_s, k_s, v_s, m_s, scale=scale, is_causal=False)
        .squeeze(2)
        .unsqueeze(0)
    )

    # TTNN decode #1 — BF16 mask (CB uses BF16 dataformat; expect 1 cache entry)
    tt_mask_bf16 = ttnn.as_tensor(
        mask_torch.transpose(1, 2).contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )
    out_bf16 = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        is_causal=False,
        attn_mask=tt_mask_bf16,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
    )
    data_bf16 = ttnn.to_torch(out_bf16)[:, :, :nh, :]
    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after first decode (BFLOAT16 mask), got {device.num_program_cache_entries()}"

    # TTNN decode #2 — BFP8 mask (expect 2nd cache entry if attn_mask is hashed; otherwise reuses BF16 program)
    tt_mask_bfp8 = ttnn.as_tensor(
        mask_torch.transpose(1, 2).contiguous(),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram,
    )
    out_bfp8 = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        is_causal=False,
        attn_mask=tt_mask_bfp8,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
    )
    data_bfp8 = ttnn.to_torch(out_bfp8)[:, :, :nh, :]
    assert device.num_program_cache_entries() == 2, (
        f"Expected 2 cache entries after second decode (BFLOAT8_B mask; compile-time mask dtype), "
        f"got {device.num_program_cache_entries()}. If this is 1, compute_program_hash likely "
        f"only uses has_attn_mask — see transformer_sdpa_decode_audit.md BUG #2."
    )

    # PCC vs golden
    min_pcc = 0.97
    for label, tt_data in (("BFLOAT16 mask", data_bf16), ("BFLOAT8_B mask", data_bfp8)):
        ok, pcc = comp_pcc(ref, tt_data, min_pcc)
        assert ok, f"{label}: output vs PyTorch PCC failed ({pcc}); possible wrong cached program."

    device.disable_and_clear_program_cache()


def _run_decode_causal_with_cur_pos_variant(
    device,
    cur_pos_dram: bool,
    torch_state: int,
    *,
    b: int = 4,
    nh: int = 8,
    nkv: int = 1,
    s: int = 1024,
    d: int = 64,
    grid_size: tuple[int, int] = (8, 4),
    verify_outputs: bool = True,
):
    compute_grid_size = device.compute_with_storage_grid_size()
    if grid_size[0] > compute_grid_size.x or grid_size[1] > compute_grid_size.y:
        pytest.skip(f"Need grid {grid_size}, device has {compute_grid_size}")

    torch.manual_seed(torch_state)

    padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
    dram = ttnn.DRAM_MEMORY_CONFIG

    shard_grid = ttnn.CoreRangeSet({num_to_corerange(b)})
    shard_spec_q = ttnn.ShardSpec(shard_grid, (padded_num_heads, d), ttnn.ShardOrientation.ROW_MAJOR)
    height_sharded_memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec_q)

    k = fa_rand(b, nkv, s, d)
    v = fa_rand(b, nkv, s, d)
    tt_k = ttnn.as_tensor(k, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_v = ttnn.as_tensor(v, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=dram)

    start_indices = [min(s // 4 + i, s - 1) for i in range(b)]
    max_start_idx = max(start_indices)
    scale = d**-0.5
    k_chunk_size = get_chunk_size(max_start_idx + 1, s)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=padded_num_heads,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    padded_layer_len = nearest_n(max_start_idx + 1, n=k_chunk_size)

    mask_torch = torch.zeros((b, nh, 1, padded_layer_len))
    for i in range(b):
        start_idx = start_indices[i]
        mask_torch[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min

    q = fa_rand(1, b, nh, d)
    tt_q = ttnn.as_tensor(
        q[:, :, :nh],
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=height_sharded_memcfg,
    )

    pos_torch = torch.tensor(start_indices, dtype=torch.int32)
    if cur_pos_dram:
        tt_pos = ttnn.from_torch(
            pos_torch,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=dram,
        )
    else:
        pos_2d = pos_torch.reshape(b, 1)
        shard_spec_pos = ttnn.ShardSpec(shard_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR)
        pos_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec_pos,
        )
        tt_pos = ttnn.from_torch(
            pos_2d,
            device=device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=pos_memcfg,
        )

    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tt_q,
        tt_k,
        tt_v,
        cur_pos_tensor=tt_pos,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=dram,
    )

    if not verify_outputs:
        return None

    tt_back = ttnn.to_torch(out)[:, :, :nh, :]

    q_s = q[:, :, :nh, :].permute(1, 2, 0, 3)
    k_s = k[:, :, :padded_layer_len, :]
    k_s = torch.cat([k_s[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    v_s = v[:, :, :padded_layer_len, :]
    v_s = torch.cat([v_s[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    m_s = mask_torch[:, :nh, :, :]
    ref = (
        torch.nn.functional.scaled_dot_product_attention(q_s, k_s, v_s, m_s, scale=scale, is_causal=False)
        .squeeze(2)
        .unsqueeze(0)
    )

    return comp_pcc(ref, tt_back, 0.97)


@pytest.mark.timeout(600)
def test_cur_pos_dram_then_sharded_smaller(device):
    """BUG #3: DRAM vs sharded ``cur_pos_tensor`` must produce two program-cache entries (no bad reuse)."""
    device.disable_and_clear_program_cache()
    device.enable_program_cache()

    kw = dict(b=2, nh=4, nkv=1, s=64, d=64, grid_size=(4, 2))

    try:
        _run_decode_causal_with_cur_pos_variant(device, cur_pos_dram=True, torch_state=203, verify_outputs=False, **kw)
    except RuntimeError as e:
        pytest.fail(f"DRAM cur_pos_tensor path failed (setup issue): {e}")

    assert (
        device.num_program_cache_entries() == 1
    ), f"Expected 1 cache entry after DRAM cur_pos decode, got {device.num_program_cache_entries()}"

    try:
        _run_decode_causal_with_cur_pos_variant(device, cur_pos_dram=False, torch_state=203, verify_outputs=False, **kw)
    except RuntimeError as e:
        pytest.skip(f"Sharded cur_pos_tensor not supported in this build/config: {e}")

    assert device.num_program_cache_entries() == 2, (
        f"Expected 2 cache entries after sharded cur_pos (distinct layout vs DRAM path), "
        f"got {device.num_program_cache_entries()} — see transformer_sdpa_decode_audit.md BUG #3."
    )

    device.disable_and_clear_program_cache()
