// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "associative_scan.hpp"
#include "device/associative_scan_device_operation.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn {
namespace operations {
namespace reduction {

ttnn::Tensor AssociativeScanOperation::invoke(
    const ttnn::Tensor& input,
    int32_t dim,
    const std::optional<MemoryConfig>& memory_config) {
    
    // We expect the input to be appropriately padded and on device.
    // In a full implementation, we'd add input validation here.
    return ttnn::prim::associative_scan(input, dim, memory_config);
}

}  // namespace reduction
}  // namespace operations
}  // namespace ttnn
