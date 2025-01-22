/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <limits>

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
    FLOAT_ACCUM sum        = 0;
    for(size_t i = 0; i < input_tv.size[1]; i++)
    {
        int64_t embedding_idx = input[input_tv.get_tensor_view_idx({output_layout.layout[0], i})];

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            FLOAT_ACCUM scale =
                per_sample_weights
                    ? CVT_FLOAT2ACCUM(per_sample_weights[per_sample_weights_tv.get_tensor_view_idx(
                          {output_layout.layout[0], i})])
                    : CVT_FLOAT2ACCUM(1);
            sum += CVT_FLOAT2ACCUM(weight[weight_tv.get_tensor_view_idx(
                       {embedding_idx, output_layout.layout[1]})]) *
                   scale;
        }
    }

    output[output_tv.get_tensor_view_idx(output_layout)] =
        (mode == 1) ? CVT_ACCUM2FLOAT(sum / input_tv.size[1]) : CVT_ACCUM2FLOAT(sum);
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
__device__ void EmbeddingBagMaxForwardKernel(const int64_t* input,
                                             const TIO* weight,
                                             TIO* output,
                                             tensor_view_t<2> input_tv,
                                             tensor_view_t<2> weight_tv,
                                             tensor_view_t<2> output_tv)
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
    FLOAT_ACCUM m       = std::numeric_limits<FLOAT_ACCUM>::min();
    for(auto i = 0; i < input_tv.size[1]; i++)
    {
        int64_t embedding_idx = input[input_tv.get_tensor_view_idx({output_layout.layout[0], i})];

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            FLOAT_ACCUM w = CVT_FLOAT2ACCUM(
                weight[weight_tv.get_tensor_view_idx({embedding_idx, output_layout.layout[1]})]);
            if(w > m)
            {
                m = w;
            }
        }
    }

    output[output_tv.get_tensor_view_idx(output_layout)] = CVT_ACCUM2FLOAT(m);
}

extern "C" __global__ void EmbeddingBagMaxForward(const int64_t* input,
                                                  const IO_TYPE* weight,
                                                  IO_TYPE* output,
                                                  tensor_view_t<2> input_tv,
                                                  tensor_view_t<2> weight_tv,
                                                  tensor_view_t<2> output_tv)
{
    EmbeddingBagMaxForwardKernel<IO_TYPE>(input, weight, output, input_tv, weight_tv, output_tv);
}
