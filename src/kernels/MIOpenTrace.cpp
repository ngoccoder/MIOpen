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

#define LOCAL_SIZE 256

template <typename TI, typename TO>
__device__ void TraceForward_kernel(
    const TI* input, TO* output, size_t N, tensor_view_t<2> input_tv, bool isReduced)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t lid = threadIdx.x;

    __shared__ FLOAT_ACCUM local_mem[LOCAL_SIZE];

    if(gid < N)
    {
        tensor_layout_t<2> diag_layout = {gid, gid};
        auto input_idx                 = input_tv.get_tensor_view_idx(diag_layout);
        local_mem[lid]                 = CVT_FLOAT2ACCUM(input[input_idx]);
    }
    else
    {
        local_mem[lid] = 0;
    }

    __syncthreads();

    for(size_t i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            local_mem[lid] += local_mem[lid + i];
        }
        __syncthreads();
    }

    if(lid == 0)
    {
        output[blockIdx.x] = isReduced ? local_mem[0] : CVT_ACCUM2FLOAT(local_mem[0]);
    }
}

extern "C" __global__ void TraceForward(
    const IO_TYPE* input, O_TYPE* output, size_t N, tensor_view_t<2> input_tv, bool isReduced)
{
    TraceForward_kernel<IO_TYPE, O_TYPE>(input, output, N, input_tv, isReduced);
}

template <typename TIO>
__device__ void TraceBackward_kernel(const TIO* output_grad,
                                     TIO* input_grad,
                                     size_t N,
                                     tensor_view_t<2> input_grad_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    size_t idx = gid % (input_grad_tv.size[1] + 1);

    if(idx != input_grad_tv.size[1])
    {
        TIO val                                                      = output_grad[0];
        tensor_layout_t<2> ingrad_layout                             = {gid, idx};
        input_grad[input_grad_tv.get_tensor_view_idx(ingrad_layout)] = val;
    }
}

extern "C" __global__ void TraceBackward(const IO_TYPE* output_grad,
                                         IO_TYPE* input_grad,
                                         size_t N,
                                         tensor_view_t<2> input_grad_tv)
{
    TraceBackward_kernel<IO_TYPE>(output_grad, input_grad, N, input_grad_tv);
}
