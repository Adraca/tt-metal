// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {
namespace prim {

operation::ProgramWithCallbacks associative_scan_multi_core(
    const Tensor& input,
    const Tensor& output,
    int32_t dim);

}  // namespace prim
}  // namespace ttnn
