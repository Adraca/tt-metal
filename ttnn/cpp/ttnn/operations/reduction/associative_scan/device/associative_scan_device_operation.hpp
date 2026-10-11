// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace prim {

struct AssociativeScanDeviceOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,
        int32_t dim,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace prim

constexpr auto associative_scan = ttnn::register_operation<"ttnn::prim::associative_scan", prim::AssociativeScanDeviceOperation>();

}  // namespace ttnn
