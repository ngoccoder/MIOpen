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
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_utils.hpp"

template <typename T>
__device__ void VarBackwardImpl(const T* __restrict__ input,
                                T* __restrict__ input_grad,
                                const T* __restrict__ mean,
                                const T* __restrict__ mean_grad,
                                const T* __restrict__ var_grad,
                                uint64_t N,
                                dim_5d_t dims,
                                bool unbiased,
                                uint64_t divisor,
                                tensor_view input_tv,
                                tensor_view input_grad_tv,
                                tensor_view mean_tv,
                                tensor_view mean_grad_tv,
                                tensor_view var_grad_tv)
{
    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lid = threadIdx.x;
    if(gid >= N)
        return;

    uint64_t n[5];
    GET_NCDHW(n, gid, input_grad_tv.dimensions);

    uint64_t o[5];
    for(int i = 0; i < 5; i++)
    {
        o[i] = dims.x[i] ? 0 : n[i];
    }

    uint64_t input_index = GET_STRIDED_INDEX(n, input_tv.strides);
    T input_v            = input[input_index];
    T mean_v             = 0;
    if(mean)
    {
        uint64_t mean_index = GET_STRIDED_INDEX(o, mean_tv.strides);
        mean_v              = mean[mean_index];
    }

    T input_grad_v = 0;
    if(var_grad)
    {
        uint64_t var_grad_index = GET_STRIDED_INDEX(o, var_grad_tv.strides);
        T var_grad_v            = var_grad[var_grad_index];
        T res                   = var_grad_v * (input_v - mean_v) * 2;
        input_grad_v += unbiased ? res / (divisor - 1) : res / divisor;
    }

    if(mean_grad)
    {
        uint64_t mean_grad_index = GET_STRIDED_INDEX(o, mean_grad_tv.strides);
        T mean_grad_v            = mean_grad[mean_grad_index];
        input_grad_v += mean_grad_v / divisor;
    }

    uint64_t input_grad_index    = GET_STRIDED_INDEX(n, input_grad_tv.strides);
    input_grad[input_grad_index] = input_grad_v;
}

template <typename T>
__device__ void VarBackwardContiguousImpl(const T* __restrict__ input,
                                          T* __restrict__ input_grad,
                                          const T* __restrict__ mean,
                                          const T* __restrict__ mean_grad,
                                          const T* __restrict__ var_grad,
                                          uint64_t N,
                                          dim_5d_t dims,
                                          bool unbiased,
                                          uint64_t divisor,
                                          tensor_view input_grad_tv,
                                          tensor_view mean_tv,
                                          tensor_view mean_grad_tv,
                                          tensor_view var_grad_tv)
{
    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lid = threadIdx.x;
    if(gid >= N)
        return;

    uint64_t n[5];
    GET_NCDHW(n, gid, input_grad_tv.dimensions);

    uint64_t o[5];
    for(int i = 0; i < 5; i++)
    {
        o[i] = dims.x[i] ? 0 : n[i];
    }

    T input_v = input[gid];
    T mean_v  = 0;
    if(mean)
    {
        uint64_t mean_index = GET_STRIDED_INDEX(o, mean_tv.strides);
        mean_v              = mean[mean_index];
    }

    T input_grad_v = 0;
    if(var_grad)
    {
        uint64_t var_grad_index = GET_STRIDED_INDEX(o, var_grad_tv.strides);
        T var_grad_v            = var_grad[var_grad_index];
        T res                   = var_grad_v * (input_v - mean_v) * 2;
        input_grad_v += unbiased ? res / (divisor - 1) : res / divisor;
    }

    if(mean_grad)
    {
        uint64_t mean_grad_index = GET_STRIDED_INDEX(o, mean_grad_tv.strides);
        T mean_grad_v            = mean_grad[mean_grad_index];
        input_grad_v += mean_grad_v / divisor;
    }

    input_grad[gid] = input_grad_v;
}

extern "C" __global__ void VarBackward(const FLOAT* __restrict__ input,
                                       FLOAT* __restrict__ input_grad,
                                       const FLOAT* __restrict__ mean,
                                       const FLOAT* __restrict__ mean_grad,
                                       const FLOAT* __restrict__ var_grad,
                                       uint64_t N,
                                       dim_5d_t dims,
                                       bool unbiased,
                                       uint64_t divisor,
                                       tensor_view input_tv,
                                       tensor_view input_grad_tv,
                                       tensor_view mean_tv,
                                       tensor_view mean_grad_tv,
                                       tensor_view var_grad_tv)
{
    VarBackwardImpl<FLOAT>(input,
                           input_grad,
                           mean,
                           mean_grad,
                           var_grad,
                           N,
                           dims,
                           unbiased,
                           divisor,
                           input_tv,
                           input_grad_tv,
                           mean_tv,
                           mean_grad_tv,
                           var_grad_tv);
}

extern "C" __global void VarBackwardContiguous(const FLOAT* __restrict__ input,
                                               FLOAT* __restrict__ input_grad,
                                               const FLOAT* __restrict__ mean,
                                               const FLOAT* __restrict__ mean_grad,
                                               const FLOAT* __restrict__ var_grad,
                                               uint64_t N,
                                               dim_5d_t dims,
                                               bool unbiased,
                                               uint64_t divisor,
                                               tensor_view input_grad_tv,
                                               tensor_view mean_tv,
                                               tensor_view mean_grad_tv,
                                               tensor_view var_grad_tv)
{
    VarBackwardContiguousImpl<FLOAT>(input,
                                     input_grad,
                                     mean,
                                     mean_grad,
                                     var_grad,
                                     N,
                                     dims,
                                     unbiased,
                                     divisor,
                                     input_grad_tv,
                                     mean_tv,
                                     mean_grad_tv,
                                     var_grad_tv);
}