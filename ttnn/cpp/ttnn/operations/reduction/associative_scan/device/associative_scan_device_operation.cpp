// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "associative_scan_device_operation.hpp"
#include "associative_scan_program_factory.hpp"
#include "tt_metal/host_api.hpp"

namespace ttnn {
namespace prim {

ttnn::Tensor AssociativeScanDeviceOperation::invoke(
    const ttnn::Tensor& input,
    int32_t dim,
    const std::optional<MemoryConfig>& memory_config) {
    
    // Allocate output tensor
    auto output = ttnn::Tensor(input.get_logical_shape(), input.get_dtype(), input.layout(), input.device(), memory_config.value_or(input.memory_config()));

    // Auto-launch the multi-core program
    auto program = associative_scan_multi_core(input, output, dim);
    tt::tt_metal::EnqueueProgram(input.device()->command_queue(), program.program, false);

    return output;
}

}  // namespace prim
}  // namespace ttnn
