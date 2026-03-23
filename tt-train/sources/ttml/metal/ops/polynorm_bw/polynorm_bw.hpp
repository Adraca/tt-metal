// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>
#include <utility>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

ttnn::Tensor polynorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float w0,
    float w1,
    float w2,
    float epsilon = 1e-5F);

// Returns [dL_dx, dL_dw, dL_db], where dL_dw has shape [1,1,1,3] and dL_db [1,1,1,1].
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> polynorm_bw_full(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float w0,
    float w1,
    float w2,
    float epsilon = 1e-5F);

// Experimental fused backward: computes dL_dx and packed param-grad partials in one kernel.
// Returns [dL_dx, dL_dw, dL_db], where dL_dw has shape [1,1,1,3] and dL_db [1,1,1,1].
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> polynorm_bw_full_fused_partials(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float w0,
    float w1,
    float w2,
    float epsilon = 1e-5F);

// Returns raw fused outputs [dL_dx, packed_partials] for stage-wise debug.
// debug_stage:
//   0 -> normal packed_partials [dw0_row, dw1_row, dw2_row, db_row]
//   1 -> stage-1 debug packed_partials [inv_rms_x, inv_rms_x2, inv_rms_x3, 0]
std::pair<ttnn::Tensor, ttnn::Tensor> polynorm_bw_fused_partials_raw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float w0,
    float w1,
    float w2,
    float epsilon = 1e-5F,
    uint32_t debug_stage = 0U);

}  // namespace ttml::metal
