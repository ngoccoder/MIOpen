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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

template <typename TIO>
__device__ void EmbeddingBagForwardKernel(const int64_t* input,
                                          const TIO* weight,
                                          TIO* output,
                                          const TIO* per_sample_weights,
                                          int32_t mode,
                                          tensor_view_t<2> input_tv,
                                          tensor_view_t<2> weight_tv,
                                          tensor_view_t<2> output_tv,
                                          tensor_view_t<2> per_sample_weights_tv)
{
    /*
     * input = (N, A)
     * weight = (num_embeddings, embedding_dim)
     * output = (N, embedding_dim)
     * gws = {ceil(N * embedding_dim, LOCAL_SIZE)}, lws = {LOCAL_SIZE}
     */
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    tensor_layout_t<2> output_layout(output_tv, gid);
    if(output_layout.layout[0] >= input_tv.size[0])
        return;

    int64_t num_embeddings = weight_tv.size[0];
    TIO sum                = 0;
    for(size_t i = 0; i < input_tv.size[1]; i++)
    {
        int64_t embedding_idx = input[input_tv.get_tensor_view_idx({output_layout.layout[0], i})];

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            TIO scale = per_sample_weights
                            ? per_sample_weights[per_sample_weights_tv.get_tensor_view_idx(
                                  {output_layout.layout[0], i})]
                            : 1;
            sum += weight[weight_tv.get_tensor_view_idx({embedding_idx, output_layout.layout[1]})] *
                   scale;
        }
    }

    output[output_tv.get_tensor_view_idx(output_layout)] =
        (mode == 1) ? sum / input_tv.size[1] : sum;
}

// for EMBEDDING_BAG_[SUM|MEAN] mode without offsets tensor
extern "C" __global__ void EmbeddingBagForward(const int64_t* input,
                                               const IO_TYPE* weight,
                                               IO_TYPE* output,
                                               const IO_TYPE* per_sample_weights,
                                               int32_t mode,
                                               tensor_view_t<2> input_tv,
                                               tensor_view_t<2> weight_tv,
                                               tensor_view_t<2> output_tv,
                                               tensor_view_t<2> per_sample_weights_tv)
{
    EmbeddingBagForwardKernel<IO_TYPE>(input,
                                       weight,
                                       output,
                                       per_sample_weights,
                                       mode,
                                       input_tv,
                                       weight_tv,
                                       output_tv,
                                       per_sample_weights_tv);
}

template <typename TIO>
__device__ void EmbeddingBagWithOffsetsForwardKernel(const int64_t* input,
                                                     const TIO* weight,
                                                     TIO* output,
                                                     const int64_t* offsets,
                                                     const TIO* per_sample_weights,
                                                     int64_t* offset2bag,
                                                     int64_t* bag_size,
                                                     int32_t mode,
                                                     tensor_view_t<1> input_tv,
                                                     tensor_view_t<2> weight_tv,
                                                     tensor_view_t<2> output_tv,
                                                     tensor_view_t<1> offsets_tv)
{
    /*
     * B = num_bags, M = num_embeddings, H = embedding_dim
     * input = (N)
     * weight = (num_embeddings, H)
     * offsets = (B)
     * output = (B, H)
     * lws = {LOCAL_SIZE}
     * gws = {AlignUp(B * H, LOCAL_SIZE)}
     */

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    tensor_layout_t<2> weight_layout(weight_tv, gid); // (bag, feature_dim)
    size_t bag         = weight_layout.layout[0];
    size_t feature_dim = weight_layout.layout[1];

    if(bag >= output_tv.size[0])
        return;

    TIO sum = 0;

    int64_t input_start    = offsets[bag];
    int64_t input_end      = (bag + 1 < offsets_tv.size[0]) ? offsets[bag + 1] : input_tv.size[0];
    int32_t divisor        = (input_end - input_start);
    int64_t num_embeddings = weight_tv.size[0];

    for(int64_t i = input_start; i < input_end; i++)
    {
        int64_t embedding_idx = input[i];

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            TIO scale = per_sample_weights ? per_sample_weights[i] : 1;
            sum += weight[weight_tv.get_tensor_view_idx({embedding_idx, feature_dim})] * scale;
        }

        if(offset2bag)
        {
            offset2bag[i - embedding_idx] = bag;
        }
    }

    output[output_tv.get_tensor_view_idx({bag, feature_dim})] =
        (mode == 1) ? (divisor ? (sum / divisor) : 0) : sum;
}

extern "C" __global__ void EmbeddingBagWithOffsetsForward(const int64_t* input,
                                                          const IO_TYPE* weight,
                                                          IO_TYPE* output,
                                                          const int64_t* offsets,
                                                          const IO_TYPE* per_sample_weights,
                                                          int64_t* offset2bag,
                                                          int64_t* bag_size,
                                                          int32_t mode,
                                                          tensor_view_t<1> input_tv,
                                                          tensor_view_t<2> weight_tv,
                                                          tensor_view_t<2> output_tv,
                                                          tensor_view_t<1> offsets_tv)
{
    EmbeddingBagWithOffsetsForwardKernel<IO_TYPE>(input,
                                                  weight,
                                                  output,
                                                  offsets,
                                                  per_sample_weights,
                                                  offset2bag,
                                                  bag_size,
                                                  mode,
                                                  input_tv,
                                                  weight_tv,
                                                  output_tv,
                                                  offsets_tv);
}

template <typename TIO>
__device__ void EmbeddingBagMaxForwardKernel(const int64_t* input,
                                             const TIO* weight,
                                             TIO* output,
                                             int64_t* max_indices,
                                             tensor_view_t<2> input_tv,
                                             tensor_view_t<2> weight_tv,
                                             tensor_view_t<2> output_tv,
                                             tensor_view_t<2> max_indices_tv)
{
    /*
     * input = (N, A)
     * weight = (num_embeddings, embedding_dim)
     * output = (N, embedding_dim)
     * gws = {ceil(N * embedding_dim, LOCAL_SIZE)}, lws = {LOCAL_SIZE}
     */

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    tensor_layout_t<2> output_layout(output_tv, gid);
    if(output_layout.layout[0] >= input_tv.size[0])
        return;

    auto num_embeddings = weight_tv.size[0];
    TIO m               = std::numeric_limits<TIO>::min();
    int64_t mi          = 0;
    for(auto i = 0; i < input_tv.size[1]; i++)
    {
        int64_t embedding_idx = input[input_tv.get_tensor_view_idx({output_layout.layout[0], i})];

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            TIO w = weight[weight_tv.get_tensor_view_idx({embedding_idx, output_layout.layout[1]})];
            if(w > m)
            {
                m  = w;
                mi = embedding_idx;
            }
        }
    }

    output[output_tv.get_tensor_view_idx(output_layout)]           = m;
    max_indices[max_indices_tv.get_tensor_view_idx(output_layout)] = mi;
}

extern "C" __global__ void EmbeddingBagMaxForward(const int64_t* input,
                                                  const IO_TYPE* weight,
                                                  IO_TYPE* output,
                                                  int64_t* max_indices,
                                                  tensor_view_t<2> input_tv,
                                                  tensor_view_t<2> weight_tv,
                                                  tensor_view_t<2> output_tv,
                                                  tensor_view_t<2> max_indices_tv)
{
    EmbeddingBagForwardKernel<IO_TYPE>(
        input, weight, output, max_indices, input_tv, weight_tv, output_tv, max_indices_tv);
}

template <typename TIO>
__device__ void EmbeddingBagMaxWithOffsetsForwardKernel(const int64_t* input,
                                                        const TIO* weight,
                                                        TIO* output,
                                                        const int64_t* offsets,
                                                        int64_t* offset2bag,
                                                        int64_t* bag_size,
                                                        int64_t* max_indices,
                                                        tensor_view_t<1> input_tv,
                                                        tensor_view_t<2> weight_tv,
                                                        tensor_view_t<2> output_tv,
                                                        tensor_view_t<1> offsets_tv,
                                                        tensor_view_t<2> max_indices_tv)
{
    /*
     * B = num_bags, M = num_embeddings, H = embedding_dim
     * input = (N)
     * weight = (num_embeddings, H)
     * offsets = (B)
     * output = (B, H)
     * max_indices = (B, H)
     * lws = {LOCAL_SIZE}
     * gws = {AlignUp(B * H, LOCAL_SIZE)}
     */
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    tensor_layout_t<2> output_layout(output_tv, gid); // (bag, feature_dim)
    size_t bag         = output_layout.layout[0];
    size_t feature_dim = output_layout.layout[1];

    if(bag >= output_tv.size[0])
        return;

    size_t input_start = offsets[bag];
    size_t input_end   = (bag + 1 < offsets_tv.size[0]) ? offsets[bag + 1] : input_tv.size[0];

    TIO m             = std::numeric_limits<TIO>::min();
    int64_t mi        = 0;
    int64_t bag_size_ = 0;

    for(size_t i = input_start; i < input_end; i++)
    {
        auto embedding_idx = input[i];

        if(embedding_idx >= 0 && embedding_idx < weight_tv.size[0])
        {
            bag_size++;
            TIO w = weight[weight_tv.get_tensor_view_idx({embedding_idx, feature_dim})];
            if(w > m)
            {
                m  = w;
                mi = input[i];
            }
            if(offset2bag)
            {
                offset2bag[i - embedding_idx] = bag;
            }
        }
    }

    if(bag_size)
    {
        bag_size[bag] = bag_size_;
    }

    output[output_tv.get_tensor_view_idx({bag, feature_dim})] =
        m == std::numeric_limits<TIO>::min() ? 0 : m;
    max_indices[max_indices_tv.get_tensor_view_idx({bag, feature_dim})] = mi;
}

extern "C" __global__ void EmbeddingBagMaxWithOffsetsForward(const int64_t* input,
                                                             const IO_TYPE* weight,
                                                             IO_TYPE* output,
                                                             const int64_t* offsets,
                                                             int64_t* offset2bag,
                                                             int64_t* bag_size,
                                                             int64_t* max_indices,
                                                             tensor_view_t<1> input_tv,
                                                             tensor_view_t<2> weight_tv,
                                                             tensor_view_t<2> output_tv,
                                                             tensor_view_t<1> offsets_tv,
                                                             tensor_view_t<2> max_indices_tv)
{
    EmbeddingBagMaxWithOffsetsForwardKernel<IO_TYPE>(input,
                                                     weight,
                                                     output,
                                                     offsets,
                                                     offset2bag,
                                                     bag_size,
                                                     max_indices,
                                                     input_tv,
                                                     weight_tv,
                                                     output_tv,
                                                     offsets_tv,
                                                     max_indices_tv);
}
