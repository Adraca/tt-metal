// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw.hpp"

#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "device/polynorm_bw_device_operation.hpp"
#include "device/polynorm_bw_fused_partials_device_operation.hpp"
#include "device/polynorm_bw_param_grads_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor polynorm_bw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& dL_dout_tensor, float w0, float w1, float w2, float epsilon) {
    return ttnn::prim::ttml_polynorm_bw(input_tensor, dL_dout_tensor, w0, w1, w2, epsilon);
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> polynorm_bw_full(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& dL_dout_tensor, float w0, float w1, float w2, float epsilon) {
    const auto dL_dx = polynorm_bw(input_tensor, dL_dout_tensor, w0, w1, w2, epsilon);
    auto packed_grads = ttnn::prim::ttml_polynorm_bw_param_grads(input_tensor, dL_dout_tensor, epsilon);
    auto packed_values = core::to_vector(packed_grads);
    constexpr uint32_t kTileWidth = 32U;
    auto dL_dw_values =
        std::vector<float>{packed_values[0U], packed_values[kTileWidth], packed_values[2U * kTileWidth]};
    auto dL_dw = core::from_vector(dL_dw_values, ttnn::Shape({1, 1, 1, 3}), &autograd::ctx().get_device());
    auto dL_db = core::from_vector(
        std::vector<float>{packed_values[3U * kTileWidth]}, ttnn::Shape({1, 1, 1, 1}), &autograd::ctx().get_device());

    return {dL_dx, dL_dw, dL_db};
}

std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> polynorm_bw_full_fused_partials(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& dL_dout_tensor, float w0, float w1, float w2, float epsilon) {
    auto [dL_dx, packed_partials] = polynorm_bw_fused_partials_raw(input_tensor, dL_dout_tensor, w0, w1, w2, epsilon);
    auto reduced_partials = ttnn::sum(
        packed_partials,
        /*dim_arg=*/ttsl::SmallVector<int>{0, 1, 2},
        /*keep_dim=*/true,
        /*output_mem_config=*/std::nullopt,
        /*compute_kernel_config=*/std::nullopt);

    constexpr uint32_t kTileWidth = 32U;
    auto reduced_values = core::to_vector(reduced_partials);
    auto dL_dw = core::from_vector(
        std::vector<float>{reduced_values[0U], reduced_values[kTileWidth], reduced_values[2U * kTileWidth]},
        ttnn::Shape({1, 1, 1, 3}),
        &autograd::ctx().get_device());
    auto dL_db = core::from_vector(
        std::vector<float>{reduced_values[3U * kTileWidth]}, ttnn::Shape({1, 1, 1, 1}), &autograd::ctx().get_device());

    return {dL_dx, dL_dw, dL_db};
}

std::pair<ttnn::Tensor, ttnn::Tensor> polynorm_bw_fused_partials_raw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float w0,
    float w1,
    float w2,
    float epsilon,
    uint32_t debug_stage) {
    auto fused_outputs = ttnn::prim::ttml_polynorm_bw_fused_partials(
        input_tensor, dL_dout_tensor, w0, w1, w2, epsilon, std::nullopt, std::nullopt, debug_stage);
    TT_FATAL(fused_outputs.size() == 2U, "Fused polynorm bw expected 2 outputs, got {}", fused_outputs.size());
    return {fused_outputs[0], fused_outputs[1]};
}

}  // namespace ttml::metal
