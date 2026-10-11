// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "associative_scan_program_factory.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn {
namespace prim {

operation::ProgramWithCallbacks associative_scan_multi_core(
    const Tensor& input,
    const Tensor& output,
    int32_t dim) {

    Program program{};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    tt_metal::Device* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    // FABLE AUDIT V2 FIX: L1 Memory Pressure
    // Tensix L1 is 16KB. 4 tiles (8KB) x 2 CBs = 16KB (Overflows L1 kernel state).
    // Reduced to 2 tiles (4KB) per CB. 8KB total leaves 8KB for kernel state and vars.
    uint32_t num_input_tiles = 2;
    uint32_t num_output_tiles = 2;

    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(
        num_input_tiles * single_tile_size, {{CB::c_in0, cb_data_format}})
        .set_page_size(CB::c_in0, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src0_config);

    tt_metal::CircularBufferConfig cb_out0_config = tt_metal::CircularBufferConfig(
        num_output_tiles * single_tile_size, {{CB::c_out0, cb_data_format}})
        .set_page_size(CB::c_out0, single_tile_size);
    auto cb_out0 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_out0_config);

    // FABLE AUDIT V2 FIX: Semaphore Deadlock
    // Turnstile barrier requires initializing to N (not 0).
    // Row barrier N = num_cores_x. Col barrier N = num_cores_y.
    uint32_t row_sync_sem_addr = tt_metal::CreateSemaphore(program, all_device_cores, num_cores_x);
    uint32_t col_sync_sem_addr = tt_metal::CreateSemaphore(program, all_device_cores, num_cores_y);

    // Setup Kernels
    auto reader_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/associative_scan/device/kernels/dataflow/reader_associative_scan_interleaved.cpp",
        all_device_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/associative_scan/device/kernels/dataflow/writer_associative_scan_interleaved.cpp",
        all_device_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto compute_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/associative_scan/device/kernels/compute/associative_scan_compute.cpp",
        all_device_cores,
        tt_metal::ComputeConfig{});

    // Configure args for all cores
    uint32_t total_cores = num_cores_x * num_cores_y;
    for (uint32_t y = 0; y < num_cores_y; ++y) {
        for (uint32_t x = 0; x < num_cores_x; ++x) {
            CoreCoord core = {x, y};
            
            uint32_t flat_core_idx = y * num_cores_x + x;
            
            // FABLE AUDIT V2 FIX: Strided I/O Bank Conflicts
            // Pass total_cores instead of a byte stride. The dataflow kernel must implement 
            // tile-based striping internally to avoid intra-core bank serialization.
            tt_metal::SetRuntimeArgs(
                program,
                reader_id,
                core,
                {input.buffer()->address(), flat_core_idx, total_cores});

            tt_metal::SetRuntimeArgs(
                program,
                writer_id,
                core,
                {output.buffer()->address(), flat_core_idx, total_cores});
                
            tt_metal::SetRuntimeArgs(
                program,
                compute_id,
                core,
                {num_cores_x, num_cores_y, row_sync_sem_addr, col_sync_sem_addr});
        }
    }

    auto override_runtime_arguments_callback = [
        reader_id, writer_id, compute_id, all_device_cores, 
        row_sync_sem_addr, col_sync_sem_addr, num_cores_x, num_cores_y
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer_address = input_tensors.at(0).buffer()->address();
        auto dst_buffer_address = output_tensors.at(0).buffer()->address();
        
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_id);
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_id);
        
        for (auto& args : reader_runtime_args) {
            args[0].at(0) = src_buffer_address;
        }
        for (auto& args : writer_runtime_args) {
            args[0].at(0) = dst_buffer_address;
        }

        // FABLE AUDIT V2 FIX: Semaphore Reset
        // Must reset semaphores back to their initial turnstile value before re-executing
        // or the second kernel invocation will immediately deadlock.
        auto device = input_tensors.at(0).device();
        for (auto core : all_device_cores) {
            tt_metal::detail::WriteToDeviceL1(device, core, row_sync_sem_addr, std::vector<uint32_t>{num_cores_x});
            tt_metal::detail::WriteToDeviceL1(device, core, col_sync_sem_addr, std::vector<uint32_t>{num_cores_y});
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace prim
}  // namespace ttnn
