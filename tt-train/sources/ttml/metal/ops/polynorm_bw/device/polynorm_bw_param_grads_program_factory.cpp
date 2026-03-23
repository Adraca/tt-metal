// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw_param_grads_program_factory.hpp"

#include <bit>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "metal/common/program_utils.hpp"

namespace {

constexpr auto kReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/dataflow/"
    "reader_polynorm_bw_param_grads_interleaved_start_id.cpp";
constexpr auto kWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/dataflow/"
    "writer_polynorm_bw_param_grads_interleaved_start_id.cpp";
constexpr auto kComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/polynorm_bw/device/kernels/compute/polynorm_bw_param_grads_kernel.cpp";

}  // namespace

namespace ttml::metal::ops::polynorm_bw::device::param_grads {

PolyNormBackwardParamGradsProgramFactory::cached_program_t PolyNormBackwardParamGradsProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    const auto& dL_dout = tensor_args.dL_dout;
    auto* device = input.device();

    tt::tt_metal::Program program{};
    const tt::DataFormat in_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(in_format == tt::DataFormat::Float16_b, "PolyNormBackwardParamGrads supports BF16 input only");
    const tt::DataFormat out_format = datatype_to_dataformat_converter(output.dtype());
    TT_FATAL(out_format == tt::DataFormat::Float32, "PolyNormBackwardParamGrads output must be FLOAT32");

    const uint32_t bfloat16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const uint32_t float32_tile_size = tt::tile_size(tt::DataFormat::Float32);

    const auto padded_tensor_shape = input.padded_shape();
    TT_FATAL(padded_tensor_shape.rank() == 4U, "Input tensor must be 4D");
    const uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;
    const uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    const uint32_t total_rows_to_process = NC * Ht;

    const auto grid = device->compute_with_storage_grid_size();
    tt::tt_metal::CoreCoord core{0, 0};
    TT_FATAL(core.x < grid.x && core.y < grid.y, "Core (0,0) is outside compute grid");
    tt::tt_metal::CoreRangeSet single_core({tt::tt_metal::CoreRange(core, core)});

    constexpr uint32_t block_size = 4U;
    constexpr uint32_t output_wt = 4U;

    create_circular_buffer(program, single_core, tt::CBIndex::c_0, in_format, bfloat16_tile_size, block_size);
    create_circular_buffer(program, single_core, tt::CBIndex::c_1, in_format, bfloat16_tile_size, block_size);
    create_circular_buffer(program, single_core, tt::CBIndex::c_2, in_format, bfloat16_tile_size, block_size);
    create_circular_buffer(program, single_core, tt::CBIndex::c_3, in_format, bfloat16_tile_size, block_size);
    create_circular_buffer(program, single_core, tt::CBIndex::c_4, in_format, bfloat16_tile_size, block_size);
    create_circular_buffer(program, single_core, tt::CBIndex::c_5, in_format, bfloat16_tile_size, block_size);
    create_circular_buffer(program, single_core, tt::CBIndex::c_6, in_format, bfloat16_tile_size, block_size);
    create_circular_buffer(program, single_core, tt::CBIndex::c_7, in_format, bfloat16_tile_size, block_size);
    create_circular_buffer(program, single_core, tt::CBIndex::c_8, in_format, bfloat16_tile_size, block_size);
    create_circular_buffer(program, single_core, tt::CBIndex::c_9, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_10, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_11, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_12, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_13, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_14, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_15, in_format, bfloat16_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_16, in_format, bfloat16_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_17, in_format, bfloat16_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_18, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_19, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_20, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_21, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_22, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(program, single_core, tt::CBIndex::c_23, tt::DataFormat::Float32, float32_tile_size, 1U);
    create_circular_buffer(
        program, single_core, tt::CBIndex::c_24, tt::DataFormat::Float32, float32_tile_size, block_size);

    auto* input_buffer = input.buffer();
    auto* dL_dout_buffer = dL_dout.buffer();
    auto* output_buffer = output.buffer();
    TT_FATAL(input_buffer != nullptr && dL_dout_buffer != nullptr && output_buffer != nullptr, "Null buffer");
    TT_FATAL(input_buffer->buffer_type() == ttnn::BufferType::DRAM, "Input must be DRAM");
    TT_FATAL(dL_dout_buffer->buffer_type() == ttnn::BufferType::DRAM, "dL_dout must be DRAM");
    TT_FATAL(output_buffer->buffer_type() == ttnn::BufferType::DRAM, "Output must be DRAM");

    std::vector<uint32_t> reader_compile_time_args{block_size, Wt};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dL_dout_buffer).append_to(reader_compile_time_args);
    std::map<std::string, std::string> defines;
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    auto reader_kernel =
        create_reader_kernel(program, single_core, reader_compile_time_args, defines, kReaderKernelPath);

    std::vector<uint32_t> writer_compile_time_args{block_size, output_wt};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    auto writer_kernel =
        create_writer_kernel(program, single_core, writer_compile_time_args, defines, kWriterKernelPath);

    std::vector<uint32_t> compute_args = {total_rows_to_process, block_size, Wt};
    auto compute_kernel = create_compute_kernel(program, single_core, compute_args, defines, kComputeKernelPath, true);

    const uint32_t scaler_fp32_bits = std::bit_cast<uint32_t>(1.0F / static_cast<float>(input.logical_shape()[-1]));
    const uint32_t eps_fp32_bits = std::bit_cast<uint32_t>(args.epsilon);

    SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {input_buffer->address(),
         dL_dout_buffer->address(),
         total_rows_to_process,
         0U,
         scaler_fp32_bits,
         eps_fp32_bits});
    SetRuntimeArgs(program, writer_kernel, core, {output_buffer->address()});

    return cached_program_t{
        std::move(program),
        {
            reader_kernel,
            writer_kernel,
            compute_kernel,
            core,
        }};
}

void PolyNormBackwardParamGradsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    const auto core = cached_program.shared_variables.core;
    auto* input_buffer = tensor_args.input.buffer();
    auto* dL_dout_buffer = tensor_args.dL_dout.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    const uint32_t total_rows_to_process = tensor_args.input.padded_shape()[0] * tensor_args.input.padded_shape()[1] *
                                           (tensor_args.input.padded_shape()[-2] / tt::constants::TILE_HEIGHT);
    const uint32_t scaler_fp32_bits =
        std::bit_cast<uint32_t>(1.0F / static_cast<float>(tensor_args.input.logical_shape()[-1]));
    const uint32_t eps_fp32_bits = std::bit_cast<uint32_t>(operation_attributes.epsilon);

    auto& rr = GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id)[core.x][core.y];
    rr[0] = input_buffer->address();
    rr[1] = dL_dout_buffer->address();
    rr[2] = total_rows_to_process;
    rr[3] = 0U;
    rr[4] = scaler_fp32_bits;
    rr[5] = eps_fp32_bits;

    auto& wr = GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id)[core.x][core.y];
    wr[0] = output_buffer->address();
}

}  // namespace ttml::metal::ops::polynorm_bw::device::param_grads
