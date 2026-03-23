# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_group_norm_large_ex_external_cb(device):
    torch.manual_seed(0)
    shape = (1, 1, 1280 * 720, 256)  # [N, 1, H*W, C]
    num_groups = 32
    eps = 1e-5

    input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    weight = torch.randn((shape[-1],), dtype=torch.bfloat16)
    bias = torch.randn((shape[-1],), dtype=torch.bfloat16)
    c = shape[-1]
    weight_4d = weight.reshape(1, 1, c // 32, 32)
    bias_4d = bias.reshape(1, 1, c // 32, 32)

    # GroupNorm golden: convert [N,1,H*W,C] -> [N,C,1,H*W], apply GN, convert back.
    input_tensor_nchw = input_tensor.permute(0, 3, 1, 2).float()
    golden = torch.nn.functional.group_norm(
        input_tensor_nchw, num_groups=num_groups, weight=weight.float(), bias=bias.float(), eps=eps
    ).permute(0, 2, 3, 1)

    input_tensor_tt = ttnn.from_torch(input_tensor, device=device, layout=ttnn.TILE_LAYOUT)
    w_tt = ttnn.from_torch(weight_4d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    b_tt = ttnn.from_torch(bias_4d, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    sharded_mem_config, grid_size = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
        device=device,
        num_channels=c,
        num_groups=num_groups,
        input_nhw=1280 * 720,
        is_height_sharded=False,
        is_row_major=False,
    )
    output_tensor_tt = ttnn.group_norm(
        input_tensor_tt,
        num_groups=num_groups,
        epsilon=eps,
        weight=w_tt,
        bias=b_tt,
        core_grid=grid_size,
        inplace=False,
        num_out_blocks=-1,
    )
    output_tensor = ttnn.to_torch(output_tensor_tt)
    assert_with_pcc(golden, output_tensor)


# ---------------------------------------------------------------------------
# Validation-failure tests for fatals added in PR #39330
# ---------------------------------------------------------------------------


class TestGroupNormValidationFailures:
    """Device-based tests that trigger TT_FATAL / TT_THROW checks in
    groupnorm.cpp and the program factories (groupnorm_*_program_factory.cpp).

    These require a Tenstorrent device; for CPU-only helper tests see
    tests/ttnn/nightly/unit_tests/operations_compute_only/fused/test_group_norm.py
    """

    def test_grid_too_large_for_height(self, device):
        """Ht < num_virtual_rows must raise (validate_dram_grid TT_THROW).
        Shape (1,1,32,256): NHW=32, Ht=1. Grid (8,8) gives num_virtual_rows=8."""
        x = ttnn.from_torch(
            torch.randn(1, 1, 32, 256, dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        with pytest.raises(RuntimeError, match="core_grid.*is invalid|Height in tiles"):
            ttnn.group_norm(x, num_groups=32, core_grid=ttnn.CoreGrid(y=8, x=8), inplace=False)

    def test_height_not_divisible_by_virtual_rows(self, device):
        """Ht % num_virtual_rows != 0 must raise (validate_dram_grid TT_THROW).
        Shape (1,1,3*32,256): NHW=96, Ht=3. Grid (8,2) with channels=256
        gives nvc=8, rows_per_y=1, num_virtual_rows=2. 3%2!=0."""
        x = ttnn.from_torch(
            torch.randn(1, 1, 3 * 32, 256, dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        with pytest.raises(RuntimeError, match="core_grid.*is invalid|divisible by num_virtual_rows"):
            ttnn.group_norm(x, num_groups=32, core_grid=ttnn.CoreGrid(y=2, x=8), inplace=False)

    def test_channels_not_divisible_by_groups(self, device):
        """num_channels % num_groups != 0 must raise (groupnorm.cpp TT_FATAL)."""
        x = ttnn.from_torch(
            torch.randn(1, 1, 32, 256, dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        with pytest.raises(RuntimeError, match="divisible by the number of groups"):
            ttnn.group_norm(x, num_groups=7, core_grid=ttnn.CoreGrid(y=1, x=1), inplace=False)

    def test_nhw_not_tile_aligned(self, device):
        """NHW not divisible by TILE_SIZE must raise (groupnorm.cpp TT_FATAL).
        Shape (1,1,48,256): NHW=48, 48%32!=0.  ROW_MAJOR keeps the unpadded shape."""
        x = ttnn.from_torch(
            torch.randn(1, 1, 48, 256, dtype=torch.bfloat16),
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        with pytest.raises(RuntimeError, match="divisible by the tile size"):
            ttnn.group_norm(x, num_groups=32, core_grid=ttnn.CoreGrid(y=1, x=1), inplace=False)

    def test_no_valid_grid_exists(self, device):
        """When no valid sub-grid can be found, validate_dram_grid must raise.
        Shape (1,1,32,320): Ht=1, channels=320, groups=32.
        nvc for x=8 is 2, rows_per_y=4, num_virtual_rows=4*8=32 > Ht=1."""
        x = ttnn.from_torch(
            torch.randn(1, 1, 32, 320, dtype=torch.bfloat16),
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        with pytest.raises(RuntimeError, match="core_grid.*is invalid|Cannot find any valid core grid"):
            ttnn.group_norm(x, num_groups=32, core_grid=ttnn.CoreGrid(y=8, x=8), inplace=False)
