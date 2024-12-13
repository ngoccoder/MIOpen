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
#include <cstdio>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

#define LOCAL_SIZE 256

template <typename TIO>
__device__ void
TraceForward_kernel(const TIO* input, FLOAT_ACCUM* workspace, size_t N, tensor_view_t<2> input_tv)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    // const size_t lid = threadIdx.x;

    //__shared__ FLOAT_ACCUM local_mem[LOCAL_SIZE];

    if(gid < N) // gid < input_tv.size[1] ?
    {
        tensor_layout_t<2> diag_layout = {gid, gid};
        auto input_idx                 = input_tv.get_tensor_view_idx(diag_layout);
        // local_mem[lid] = static_cast<FLOAT_ACCUM>(input[input_idx]);
        workspace[gid] = static_cast<FLOAT_ACCUM>(input[input_idx]);
        // printf("local_mem[%d] = %f\n", lid, static_cast<float>(local_mem[lid]));
    }
    else
    {
        // local_mem[lid] = 0;
        workspace[gid] = 0;
    }

    /*
    __syncthreads();

    for (size_t i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (lid < i)
        {
            local_mem[lid] += local_mem[lid + i];
        }
        __syncthreads();
    }

    if (lid == 0) {
        workspace[blockIdx.x] = local_mem[0];
        printf("[kernel at block %d]N = %d\n", blockIdx.x, N);
        printf("workspace[%d] = %f\n", blockIdx.x, static_cast<float>(workspace[blockIdx.x]));
    }
    */
}

extern "C" __global__ void
TraceForward(const IO_TYPE* input, FLOAT_ACCUM* workspace, size_t N, tensor_view_t<2> input_tv)
{
    TraceForward_kernel<IO_TYPE>(input, workspace, N, input_tv);
}

template <typename TIO>
__device__ void TraceBackward_kernel(const TIO* output_grad,
                                     TIO* input_grad,
                                     size_t N,
                                     tensor_view_t<1> output_grad_tv,
                                     tensor_view_t<2> input_grad_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    size_t idx = gid % (input_grad_tv.size[1] + 1);

    if(idx != input_grad_tv.size[1])
    {
        tensor_layout_t<1> outgrad_layout = {0};
        TIO val = output_grad[output_grad_tv.get_tensor_view_idx(outgrad_layout)];
        tensor_layout_t<2> ingrad_layout                             = {gid, idx};
        input_grad[input_grad_tv.get_tensor_view_idx(ingrad_layout)] = val;
    }
}

extern "C" __global__ void TraceBackward(const IO_TYPE* output_grad,
                                         IO_TYPE* input_grad,
                                         size_t N,
                                         tensor_view_t<1> output_grad_tv,
                                         tensor_view_t<2> input_grad_tv)
{
    TraceBackward_kernel<IO_TYPE>(output_grad, input_grad, N, output_grad_tv, input_grad_tv);
}
