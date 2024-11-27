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

#include "tensor_view.hpp"

template <typename TIO, typename TINDEX>
__device__ void GatherForwardKernel(const TIO* input,
                                    const TINDEX* indices,
                                    TIO* output,
                                    uint32_t dim,
                                    tensor_view_t<4> input_tv,
                                    tensor_view_t<4> indices_tv,
                                    tensor_view_t<4> output_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t n[4], n012, n01;
    n[3] = gid % output_tv.size[3];
    n012 = gid / output_tv.size[3];
    n[2] = n012 % output_tv.size[2];
    n01  = n012 / output_tv.size[2];
    n[1] = n01 % output_tv.size[1];
    n[0] = n01 / output_tv.size[1];

    if(n[0] >= output_tv.size[0])
        return;

    size_t output_idx = output_tv.get_tensor_view_idx(tensor_layout_t<4>{n[0], n[1], n[2], n[3]});
    n[dim] = indices[indices_tv.get_tensor_view_idx(tensor_layout_t<4>{n[0], n[1], n[2], n[3]})];

    if(n[dim] >= input_tv.size[dim])
        return;

    size_t input_idx   = input_tv.get_tensor_view_idx(tensor_layout_t<4>{n[0], n[1], n[2], n[3]});
    output[output_idx] = input[input_idx]; // recheck tensor view
}

extern "C" __global__ void GatherForward(const IO_TYPE* input,
                                         const INDEX_TYPE* indices,
                                         const IO_TYPE* output,
                                         uint32_t dim,
                                         tensor_view_t<4> input_tv,
                                         tensor_view_t<4> indices_tv,
                                         tensor_view_t<4> output_tv)
{
    GatherForwardKernel<IO_TYPE, INDEX_TYPE>(
        input, indices, output, dim, input_tv, indices_tv, output_tv);
}