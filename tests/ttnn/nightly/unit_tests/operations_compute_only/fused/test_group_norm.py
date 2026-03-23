# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CPU-only host-side validation tests covering helpers introduced by PR #39330.

These tests call the C++ helpers exposed to Python (``create_group_norm_input_mask``,
``_compute_num_virtual_cols``, ``_find_expected_dram_grid``) and verify that they
raise or return correctly for valid and invalid inputs.  They do NOT require a
Tenstorrent device and are intended to run on a CPU-only CI runner.

Device-side fatals (e.g. ``Height in tiles`` from the program factories, and
``validate_dram_grid`` in ``groupnorm.cpp``) require on-device tensors and are
tested in ``tests/ttnn/nightly/unit_tests/operations/fused/test_group_norm.py``.
"""

import pytest

import ttnn
from ttnn._ttnn.operations.normalization import (
    create_group_norm_input_mask,
    _compute_num_virtual_cols,
    _find_expected_dram_grid,
)

TILE = 32


# ---------------------------------------------------------------------------
# create_group_norm_input_mask  –  validation failure tests
# ---------------------------------------------------------------------------


class TestCreateGroupNormInputMask:
    """TT_FATAL checks added to create_group_norm_input_mask_impl (PR #39330)."""

    def test_num_cores_across_channel_zero(self):
        """num_cores_across_channel == 0 must raise."""
        with pytest.raises(RuntimeError, match="num_cores_across_channel must be > 0"):
            create_group_norm_input_mask(
                num_channel=256,
                num_groups=32,
                num_cores_across_channel=0,
                data_type=ttnn.DataType.BFLOAT16,
            )

    @pytest.mark.parametrize(
        "num_groups, num_cores_across_channel",
        [
            (32, 3),
            (32, 5),
            (32, 6),
            (32, 7),
            (16, 3),
            (16, 5),
        ],
    )
    def test_num_groups_not_divisible_by_num_cores(self, num_groups, num_cores_across_channel):
        """num_groups must be divisible by num_cores_across_channel."""
        with pytest.raises(RuntimeError, match="must be divisible by num_cores_across_channel"):
            create_group_norm_input_mask(
                num_channel=256,
                num_groups=num_groups,
                num_cores_across_channel=num_cores_across_channel,
                data_type=ttnn.DataType.BFLOAT16,
            )


# ---------------------------------------------------------------------------
# _compute_num_virtual_cols  –  helper tests
# ---------------------------------------------------------------------------


class TestComputeNumVirtualCols:
    """Verify the C++ compute_num_virtual_cols helper (PR #39330)."""

    @pytest.mark.parametrize(
        "grid_x, num_groups, num_channels, expected",
        [
            (8, 32, 320, 2),
            (8, 32, 256, 8),
            (8, 32, 1280, 8),
            (1, 32, 256, 1),
            (4, 32, 256, 4),
        ],
    )
    def test_known_valid_inputs(self, grid_x, num_groups, num_channels, expected):
        result = _compute_num_virtual_cols(grid_x, num_groups, num_channels)
        assert result == expected

    def test_returns_zero_when_impossible(self):
        """If no nvc in [1, min(grid_x, num_groups)] satisfies the tile
        constraint, the function must return 0."""
        result = _compute_num_virtual_cols(grid_x=8, num_groups=32, num_channels=33)
        assert result == 0


# ---------------------------------------------------------------------------
# _find_expected_dram_grid  –  failure + correctness tests
# ---------------------------------------------------------------------------


class TestFindExpectedDramGrid:
    """Verify the C++ find_expected_dram_grid helper (PR #39330)."""

    def test_raises_when_no_valid_grid(self):
        """Impossibly small spatial dim should make the function raise."""
        with pytest.raises(RuntimeError, match="Cannot find a valid DRAM group-norm grid"):
            _find_expected_dram_grid(
                max_x=8,
                max_y=8,
                num_channels=33,
                num_groups=3,
                input_nhw=32,
            )

    @pytest.mark.parametrize(
        "max_x, max_y, num_channels, num_groups, input_nhw",
        [
            (8, 8, 320, 32, 1 * 32 * 32),
            (8, 8, 1280, 32, 1 * 16 * 16),
            (8, 8, 256, 32, 1 * 64 * 64),
            (8, 8, 480, 16, 1 * 8 * 8),
        ],
    )
    def test_valid_grid_found(self, max_x, max_y, num_channels, num_groups, input_nhw):
        grid = _find_expected_dram_grid(max_x, max_y, num_channels, num_groups, input_nhw)
        assert grid.x >= 1
        assert grid.y >= 1
        assert grid.x <= max_x
        assert grid.y <= max_y

    def test_grid_satisfies_constraints(self):
        """The returned grid must satisfy Ht >= num_virtual_rows and
        Ht % num_virtual_rows == 0 (same invariant enforced by the factories)."""
        num_channels = 320
        num_groups = 32
        input_nhw = 1 * 32 * 32

        grid = _find_expected_dram_grid(8, 8, num_channels, num_groups, input_nhw)
        nvc = _compute_num_virtual_cols(grid.x, num_groups, num_channels)
        assert nvc > 0

        rows_per_y = grid.x // nvc
        num_virtual_rows = rows_per_y * grid.y
        Ht = input_nhw // TILE
        assert Ht >= num_virtual_rows
        assert Ht % num_virtual_rows == 0
