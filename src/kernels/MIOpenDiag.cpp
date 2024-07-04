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

template <typename TI, typename TO>
__device__ void
Diag1dForwardKernel(const TI* input, TO* output, long N, long offset, tensor_view_t<2> output_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    long stride0 = output_tv.stride[0];
    long stride1 = output_tv.stride[1];

    long outputIdx    = gid * (stride0 + stride1) + offset;
    output[outputIdx] = input[gid];
}

extern "C" __global__ void Diag1dForward(
    const INPUT_TYPE* input, OUTPUT_TYPE* output, long N, long offset, tensor_view_t<2> output_tv)
{
    Diag1dForwardKernel<INPUT_TYPE, OUTPUT_TYPE>(input, output, N, offset, output_tv);
}

template <typename TI, typename TO>
__device__ void
Diag2dForwardKernel(const TI* input, TO* output, long N, long offset, tensor_view_t<2> input_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    long input_stride_0 = input_tv.stride[0];
    long input_stride_1 = input_tv.stride[1];

    long input_idx = gid * (input_stride_0 + input_stride_1) + offset;
    output[gid]    = input[input_idx];
}

extern "C" __global__ void Diag2dForward(
    const INPUT_TYPE* input, OUTPUT_TYPE* output, long N, long offset, tensor_view_t<2> input_tv)
{
    Diag2dForwardKernel<INPUT_TYPE, OUTPUT_TYPE>(input, output, N, offset, input_tv);
}

template <typename TI, typename TO>
__device__ void Assign5dKernel(
    const TI* input, TO* output, long N, tensor_view_t<5> input_tv, tensor_view_t<5> output_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    tensor_layout_t<5> inputLayout(input_tv, gid);
    auto inputIdx = input_tv.get_tensor_view_idx(inputLayout);

    tensor_layout_t<5> outputLayout(output_tv, gid);
    auto outputIdx = output_tv.get_tensor_view_idx(outputLayout);

    output[outputIdx] = input[inputIdx];
}

extern "C" __global__ void Assign5d(const INPUT_TYPE* input,
                                    OUTPUT_TYPE* output,
                                    long N,
                                    tensor_view_t<5> input_tv,
                                    tensor_view_t<5> output_tv)
{
    Assign5dKernel(input, output, N, input_tv, output_tv);
}
