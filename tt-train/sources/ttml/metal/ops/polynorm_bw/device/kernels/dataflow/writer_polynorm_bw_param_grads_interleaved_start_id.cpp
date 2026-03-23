// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t arg_idx = 0U;
    const uint32_t output_address = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto cb_output = tt::CBIndex::c_24;
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);

    const uint32_t tile_bytes = get_tile_size(cb_output);
    constexpr auto output_args = TensorAccessorArgs<2>();
    const auto output_addr_generator = TensorAccessor(output_args, output_address, tile_bytes);

    write_full_row_tiles(cb_output, output_addr_generator, Wt, block_size, tile_bytes, /*row_start_idx*/ 0U);
}
