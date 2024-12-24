/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "hip_atomic.hpp"
#include "tensor_view.hpp"
#include <cstdint>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

template <typename TIO, typename TID>
__device__ void EmbeddingBackwardKernel(const TID* input,
                                        const TIO* output_grad,
                                        TIO* weight_grad,
                                        int* indices_freq,
                                        size_t embedding_dim,
                                        size_t input_size,
                                        tensor_view_t<4> input_tv,
                                        tensor_view_t<4> output_grad_tv,
                                        tensor_view_t<2> weight_grad_tv,
                                        long embedding_base_idx,
                                        long num_embeddings,
                                        long padding_idx)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= embedding_dim)
        return;

    for(size_t i = 0; i < input_size; i++)
    {
        size_t n3   = i % input_tv.size[3];
        size_t n012 = i / input_tv.size[3];
        size_t n2   = n012 % input_tv.size[2];
        size_t n01  = n012 / input_tv.size[2];
        size_t n1   = n01 % input_tv.size[1];
        size_t n0   = n01 / input_tv.size[1];

        tensor_layout_t<4> input_layout = {n0, n1, n2, n3};
        size_t input_idx                = input_tv.get_tensor_view_idx(input_layout);
        TID embedding_idx               = input[input_idx];

        if(embedding_idx == padding_idx)
            continue;
        embedding_idx -= embedding_base_idx;

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            TIO scale =
                indices_freq ? (static_cast<TIO>(1) / static_cast<TIO>(indices_freq[i])) : 1.0;
            tensor_layout_t<2> weight_grad_layout = {embedding_idx, gid};
            size_t weight_grad_idx = weight_grad_tv.get_tensor_view_idx(weight_grad_layout);
            tensor_layout_t<4> output_grad_layout(output_grad_tv, embedding_dim * i + gid);
            weight_grad[weight_grad_idx] = weight_grad[weight_grad_idx] + output
        }
    }
}

template <typename TIO>
__device__ void EmbeddingBackwardContiguousAtomicKernel(const int64_t* input,
                                                        const TIO* output_grad,
                                                        TIO* weight_grad,
                                                        const int32_t* indices_freq,
                                                        size_t embedding_dim,
                                                        size_t input_size,
                                                        int64_t embedding_base_idx,
                                                        int64_t num_embeddings,
                                                        int64_t padding_idx)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = gid / embedding_dim, j = gid % embedding_dim;
    if(i >= input_size)
        return;

    int64_t embedding_idx = input[i];

    if(embedding_idx == padding_idx)
        return;
    embedding_idx -= embedding_base_idx;

    if(embedding_idx >= 0 && embedding_idx < num_embeddings)
    {
        TIO scale =
            indices_freq ? (static_cast<TIO>(1.0f) / static_cast<TIO>(indices_freq[i])) : 1.0f;
        atomic_add_g(&weight_grad[embedding_idx * embedding_dim + j],
                     output_grad[embedding_dim * i + j] * scale);
    }
}

extern "C" __global__ void EmbeddingBackwardContiguousAtomic(const int64_t* input,
                                                             const IO_TYPE* output_grad,
                                                             IO_TYPE* weight_grad,
                                                             const int32_t* indices_freq,
                                                             size_t embedding_dim,
                                                             size_t input_size,
                                                             int64_t embedding_base_idx,
                                                             int64_t num_embeddings,
                                                             int64_t padding_idx)
{
    EmbeddingBackwardContiguousAtomicKernel<IO_TYPE>(input,
                                                     output_grad,
                                                     weight_grad,
                                                     indices_freq,
                                                     embedding_dim,
                                                     input_size,
                                                     embedding_base_idx,
                                                     num_embeddings,
                                                     padding_idx);
}

template <typename TIO>
__device__ void EmbeddingBackwardAtomicKernel(const int64_t* input,
                                              const TIO* output_grad,
                                              TIO* weight_grad,
                                              const int32_t* indices_freq,
                                              size_t embedding_dim,
                                              size_t input_size,
                                              int64_t embedding_base_idx,
                                              int64_t num_embeddings,
                                              int64_t padding_idx,
                                              tensor_view_t<4> input_tv,
                                              tensor_view_t<4> output_grad_tv,
                                              tensor_view_t<2> weight_grad_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = gid / embedding_dim, j = gid % embedding_dim;
    if(i >= input_size)
        return;

    size_t n3 = i % input_tv.size[3], n012 = i / input_tv.size[3];
    size_t n2 = n012 % input_tv.size[2], n01 = n012 / input_tv.size[2];
    size_t n1 = n01 % input_tv.size[1], n0 = n01 / input_tv.size[1];

    tensor_layout_t<4> input_layout = {n0, n1, n2, n3};
    size_t input_idx                = input_tv.get_tensor_view_idx(input_layout);
    int64_t embedding_idx           = input[input_idx];

    if(embedding_idx == padding_idx)
        return;
    embedding_idx -= embedding_base_idx;

    if(embedding_idx >= 0 && embedding_idx < num_embeddings)
    {
        TIO scale =
            indices_freq ? (static_cast<TIO>(1.0f) / static_cast<TIO>(indices_freq[i])) : 1.0f;
        tensor_layout_t<2> weight_grad_layout = {embedding_idx, j};
        size_t weight_grad_idx = weight_grad_tv.get_tensor_view_idx(weight_grad_layout);
        tensor_layout_t<4> output_grad_layout(output_grad_tv, embedding_dim * i + j);
        atomic_add_g(&weight_grad[weight_grad_idx],
                     output_grad[output_grad_tv.get_tensor_view_idx(output_grad_layout)] * scale);
    }
}

extern "C" __global__ void EmbeddingBackwardAtomic(const int64_t* input,
                                                   const IO_TYPE* output_grad,
                                                   IO_TYPE* weight_grad,
                                                   const int32_t* indices_freq,
                                                   size_t embedding_dim,
                                                   size_t input_size,
                                                   int64_t embedding_base_idx,
                                                   int64_t num_embeddings,
                                                   int64_t padding_idx,
                                                   tensor_view_t<4> input_tv,
                                                   tensor_view_t<4> output_grad_tv,
                                                   tensor_view_t<2> weight_grad_tv)
{
    EmbeddingBackwardAtomicKernel<IO_TYPE>(input,
                                           output_grad,
                                           weight_grad,
                                           indices_freq,
                                           embedding_dim,
                                           input_size,
                                           embedding_base_idx,
                                           num_embeddings,
                                           padding_idx,
                                           input_tv,
                                           output_grad_tv,
                                           weight_grad_tv);
}

template <typename TIO>
__device__ void
EmbeddingBackwardSmallNumEmbeddingsTraverseContiguousKernel(const int64_t* input,
                                                            const TIO* output_grad,
                                                            TIO* weight_grad,
                                                            const int32_t* indices_freq,
                                                            size_t embedding_dim,
                                                            size_t input_size,
                                                            int64_t embedding_base_idx,
                                                            int64_t num_embeddings,
                                                            int64_t padding_idx,
                                                            int32_t alpha)
{
    size_t gid                   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t embedding_size        = num_embeddings * embedding_dim;
    size_t i                     = gid / embedding_size;
    size_t inner_embedding_space = gid % embedding_size;
    int32_t target_embedding_idx = inner_embedding_space / embedding_dim;
    int j                        = inner_embedding_space % embedding_dim;
    if(i >= input_size)
        return;

    TIO weight_grad_sum = 0;

    for(; i < input_size; i += alpha)
    {
        int64_t embedding_idx = input[i];
        if(embedding_idx == padding_idx)
            continue;
        embedding_idx -= embedding_base_idx;
        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            if(embedding_idx == target_embedding_idx)
            {
                TIO scale = indices_freq
                                ? (static_cast<TIO>(1.0f) / static_cast<TIO>(indices_freq[i]))
                                : 1.0f;
                weight_grad_sum += output_grad[i * embedding_dim + j] * scale;
            }
        }
    }

    atomic_add_g(&weight_grad[target_embedding_idx * embedding_dim + j], weight_grad_sum);
}

extern "C" __global__ void
EmbeddingBackwardSmallNumEmbeddingsTraverseContiguous(const int64_t* input,
                                                      const IO_TYPE* output_grad,
                                                      IO_TYPE* weight_grad,
                                                      const int32_t* indices_freq,
                                                      size_t embedding_dim,
                                                      size_t input_size,
                                                      int64_t embedding_base_idx,
                                                      int64_t num_embeddings,
                                                      int64_t padding_idx,
                                                      int32_t alpha)
{
    EmbeddingBackwardSmallNumEmbeddingsTraverseContiguousKernel<IO_TYPE>(input,
                                                                         output_grad,
                                                                         weight_grad,
                                                                         indices_freq,
                                                                         embedding_dim,
                                                                         input_size,
                                                                         embedding_base_idx,
                                                                         num_embeddings,
                                                                         padding_idx,
                                                                         alpha);
}

template <typename TIO>
__device__ void EmbeddingBackwardSmallNumEmbeddingsTraverseKernel(const int64_t* input,
                                                                  const TIO* output_grad,
                                                                  TIO* weight_grad,
                                                                  const int32_t* indices_freq,
                                                                  size_t embedding_dim,
                                                                  size_t input_size,
                                                                  int64_t embedding_base_idx,
                                                                  int64_t num_embeddings,
                                                                  int64_t padding_idx,
                                                                  int32_t alpha,
                                                                  tensor_view_t<4> input_tv,
                                                                  tensor_view_t<4> output_grad_tv,
                                                                  tensor_view_t<2> weight_grad_tv)
{
    size_t gid                   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t embedding_size        = num_embeddings * embedding_dim;
    size_t i                     = gid / embedding_size;
    size_t inner_embedding_space = gid % embedding_size;
    int32_t target_embedding_idx = inner_embedding_space / embedding_dim;
    int j                        = inner_embedding_space % embedding_dim;
    if(i >= input_size)
        return;

    TIO weight_grad_sum = 0;

    for(; i < input_size; i += alpha)
    {
        size_t n3 = i % input_tv.size[3], n012 = i / input_tv.size[3];
        size_t n2 = n012 % input_tv.size[2], n01 = n012 / input_tv.size[2];
        size_t n1 = n01 % input_tv.size[1], n0 = n01 / input_tv.size[1];

        size_t input_idx      = input_tv.get_tensor_view_idx({n0, n1, n2, n3});
        int64_t embedding_idx = input[input_idx];

        if(embedding_idx == padding_idx)
            continue;
        embedding_idx -= embedding_base_idx;
        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            if(embedding_idx == target_embedding_idx)
            {
                TIO scale = indices_freq
                                ? (static_cast<TIO>(1.0f) / static_cast<TIO>(indices_freq[i]))
                                : 1.0f;
                tensor_layout_t<4> output_grad_layout(output_grad_tv, embedding_dim * i + j);
                weight_grad_sum +=
                    output_grad[output_grad_tv.get_tensor_view_idx(output_grad_layout)] * scale;
            }
        }
    }

    atomic_add_g(&weight_grad[weight_grad_tv.get_tensor_view_idx({target_embedding_idx, j})],
                 weight_grad_sum);
}
