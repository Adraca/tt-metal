// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace operations {
namespace reduction {

struct AssociativeScanOperation {
    // Computes the associative scan of `input` along `dim`.
    // For Mamba, `input` typically contains stacked (A, B) matrices.
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,
        int32_t dim,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace reduction
}  // namespace operations

constexpr auto associative_scan =
    ttnn::register_operation_with_auto_launch_op<"ttnn::associative_scan", operations::reduction::AssociativeScanOperation>();

}  // namespace ttnn
