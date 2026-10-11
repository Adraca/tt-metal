# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_atanh(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, required_grad=True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.atanh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.atanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


def test_bw_atanh_overflow_sign_regression(device):
    # Regression for a sign inversion at grad/(1 - input^2) overflow.
    #
    # atanh_bw forces the result to +inf at the |input| == 1 singularity (where the
    # unary chain's own RECIP(-0) would otherwise return the wrong-signed -inf), then
    # used to re-derive the sign for a negative gradient by testing whether the *output*
    # equaled +inf. That test can't distinguish the singularity's forced +inf from a
    # genuine +inf produced elsewhere by grad/(1 - input^2) overflowing float32 range —
    # which happens for any sufficiently negative gradient once |input| is just past 1,
    # since 1 - input^2 is then a small negative number and the quotient is positive.
    # The old guard flipped that correct +inf to -inf. See tt-metal#54695.
    shape = torch.Size([1, 1, 32, 32])

    # 1.0078125 is the bfloat16 value immediately above 1.0 (epsilon = 2**-7 there) —
    # the point nearest the |input| == 1 pole where 1 - input^2 is smallest in magnitude
    # without being exactly zero, so grad/(1 - input^2) overflows at the smallest
    # |grad|. Mixed with its negation to cover both sides of the pole.
    x_above_one = 1.0078125
    in_data = torch.tensor(
        [x_above_one, -x_above_one] * (32 * 32 // 2), dtype=torch.bfloat16, requires_grad=True
    ).reshape(shape)

    # Large-magnitude negative gradients, well past the ~5.296e36 threshold at which
    # grad/(1 - input^2) overflows float32 at x_above_one.
    grad_data = torch.tensor([-1e38, -float("inf")] * (32 * 32 // 2), dtype=torch.bfloat16).reshape(shape)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output_tensor_on_device = ttnn.atanh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.atanh_bw)
    golden_tensor = golden_function(grad_data, in_data)[0]

    tt_out = ttnn.to_torch(tt_output_tensor_on_device[0])

    # The whole point of this case is the sign of the overflow, so check it exactly
    # rather than folding it into an aggregate PCC comparison — a small minority of
    # sign-flipped elements (as here) does not move PCC enough to fail on its own,
    # which is how this shipped in the first place.
    assert torch.equal(torch.isposinf(tt_out), torch.isposinf(golden_tensor))
    assert not torch.any(torch.isneginf(tt_out) & torch.isposinf(golden_tensor))
