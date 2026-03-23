// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw_param_grads_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::polynorm_bw::device::param_grads {

void PolyNormBackwardParamGradsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "PolyNormBackwardParamGrads operation requires {} to be on Device. Input storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to PolyNormBackwardParamGrads need device buffers. Buffer is null. Tensor name {}",
            name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "PolyNormBackwardParamGrads operation requires Tile layout. {} tensor layout: {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "PolyNormBackwardParamGrads operation requires BF16 data type. {} tensor data type: {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "PolyNormBackwardParamGrads operation requires Interleaved memory layout. {} memory layout: `{}`",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    check_tensor(tensor_args.input, "Input");
    check_tensor(tensor_args.dL_dout, "dL_dout");
    if (tensor_args.preallocated_packed_grads.has_value()) {
        const auto& out = tensor_args.preallocated_packed_grads.value();
        TT_FATAL(out.layout() == tt::tt_metal::Layout::TILE, "Preallocated packed grads must be tile layout");
        TT_FATAL(out.dtype() == tt::tt_metal::DataType::FLOAT32, "Preallocated packed grads must be FLOAT32");
    }
}

spec_return_value_t PolyNormBackwardParamGradsDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_packed_grads.has_value()) {
        return {tensor_args.preallocated_packed_grads->tensor_spec()};
    }

    auto out_mem_cfg = tensor_args.input.memory_config();
    return {ttnn::TensorSpec(
        ttnn::Shape({1, 1, 1, 128}),
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, out_mem_cfg))};
}

tensor_return_value_t PolyNormBackwardParamGradsDeviceOperation::create_output_tensors(
    const operation_attributes_t& op_attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_packed_grads.has_value()) {
        return tensor_args.preallocated_packed_grads.value();
    }
    auto specs = compute_output_specs(op_attrs, tensor_args);
    return create_device_tensor(specs[0], tensor_args.input.device());
}

ttsl::hash::hash_t PolyNormBackwardParamGradsDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return tt::tt_metal::operation::hash_operation<PolyNormBackwardParamGradsDeviceOperation>(
        args, input.dtype(), input.logical_shape());
}

}  // namespace ttml::metal::ops::polynorm_bw::device::param_grads

namespace ttnn::prim {

ttml::metal::ops::polynorm_bw::device::param_grads::PolyNormBackwardParamGradsDeviceOperation::tensor_return_value_t
ttml_polynorm_bw_param_grads(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_packed_grads) {
    using OperationType = ttml::metal::ops::polynorm_bw::device::param_grads::PolyNormBackwardParamGradsDeviceOperation;

    const auto operation_attributes = OperationType::operation_attributes_t{.epsilon = epsilon};
    const auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .dL_dout = dL_dout_tensor,
        .preallocated_packed_grads = preallocated_packed_grads,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
