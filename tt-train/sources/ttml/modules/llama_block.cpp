// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_block.hpp"

#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/grouped_query_attention.hpp"
#include "ops/binary_ops.hpp"
#include "ops/polynorm_op.hpp"
#include "ops/rope_op.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {
LlamaMLP::LlamaMLP(uint32_t embedding_size, std::optional<uint32_t> intermediate_dim, float dropout_prob) {
    uint32_t multiple_of = 256;
    uint32_t hidden_size = 0U;
    if (intermediate_dim) {
        hidden_size = *intermediate_dim;
    } else {
        const uint32_t unrounded_size = static_cast<uint32_t>(static_cast<float>(4 * embedding_size) * (2.0F / 3.0F));
        hidden_size = ((unrounded_size + multiple_of - 1U) / multiple_of) * multiple_of;
    }
    m_w1 = std::make_shared<LinearLayer>(embedding_size, hidden_size, /*has_bias=*/false);
    m_w3 = std::make_shared<LinearLayer>(embedding_size, hidden_size, /*has_bias=*/false);
    m_w2 = std::make_shared<LinearLayer>(hidden_size, embedding_size, /*has_bias=*/false);
    m_dropout = std::make_shared<DropoutLayer>(dropout_prob);
    m_polynorm_weight = autograd::create_tensor(
        core::from_vector(
            std::vector<float>{1.0F / 3.0F, 1.0F / 3.0F, 1.0F / 3.0F},
            ttnn::Shape({1, 1, 1, 3}),
            &autograd::ctx().get_device()),
        /*requires_grad=*/true);
    m_polynorm_bias = autograd::create_tensor(
        core::from_vector(std::vector<float>{0.0F}, ttnn::Shape({1, 1, 1, 1}), &autograd::ctx().get_device()),
        /*requires_grad=*/true);

    create_name("llama_mlp");
    register_module(m_w1, "w1");
    register_module(m_w3, "w3");
    register_module(m_w2, "w2");
    register_module(m_dropout, "dropout");
    register_tensor(m_polynorm_weight, "polynorm_weight");
    register_tensor(m_polynorm_bias, "polynorm_bias");
}

autograd::TensorPtr LlamaMLP::operator()(const autograd::TensorPtr& input) {
    auto swished = ops::polynorm3((*m_w1)(input), m_polynorm_weight, m_polynorm_bias);
    auto gate = (*m_w3)(input);
    auto gated = ops::mul(swished, gate);
    auto x = (*m_w2)(gated);
    x = (*m_dropout)(x);
    return x;
}

LlamaBlock::LlamaBlock(
    uint32_t embedding_size,
    uint32_t num_heads,
    uint32_t num_groups,
    const ops::RotaryEmbeddingParams& rope_params,
    float dropout_prob,
    std::optional<uint32_t> intermediate_dim) {
    m_mlp = std::make_shared<LlamaMLP>(embedding_size, intermediate_dim, dropout_prob);
    m_attention_norm = std::make_shared<RMSNormLayer>(embedding_size);
    m_mlp_norm = std::make_shared<RMSNormLayer>(embedding_size);
    m_attention = std::make_shared<GroupedQueryAttention>(GQAConfig{
        .embedding_dim = embedding_size,
        .num_heads = num_heads,
        .num_groups = num_groups,
        .dropout_prob = dropout_prob,
        .rope_params = rope_params,
        .bias_linears = false,
    });

    create_name("llama_block");
    register_module(m_mlp, "mlp");
    register_module(m_attention_norm, "attention_norm");
    register_module(m_mlp_norm, "mlp_norm");
    register_module(m_attention, "attention");
}

autograd::TensorPtr LlamaBlock::operator()(
    const autograd::TensorPtr& input, const std::optional<autograd::TensorPtr>& mask) {
    auto residual = input;
    auto h = (*m_attention_norm)(input);
    h = (*m_attention)(h, mask);  // TODO: pass in start_pos, freqs_cis for RoPE here
    h = ops::add(h, residual);

    residual = h;
    auto x = (*m_mlp_norm)(h);
    x = (*m_mlp)(x);
    x = ops::add(x, residual);

    return x;
}

autograd::TensorPtr LlamaBlock::operator()(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& mask,
    std::shared_ptr<ttml::models::common::transformer::KvCache> kv_cache,
    const uint32_t layer_idx,
    const uint32_t new_tokens) {
    auto residual = input;
    auto h = (*m_attention_norm)(input);
    h = (*m_attention)(h, mask, kv_cache, layer_idx, new_tokens);
    h = ops::add(h, residual);

    residual = h;
    auto x = (*m_mlp_norm)(h);
    x = (*m_mlp)(x);
    x = ops::add(x, residual);

    return x;
}

}  // namespace ttml::modules
