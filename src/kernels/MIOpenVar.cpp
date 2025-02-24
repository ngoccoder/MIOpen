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
__device__ void VarBackwardImpl(const TIO* input,
                                TIO* input_grad,
                                const TIO* mean,
                                const TIO* mean_grad,
                                const TIO* var_grad,
                                uint64_t N,
                                dim_5d_t dims,
                                bool unbiased,
                                uint32_t divisor,
                                tensor_view_t<5> input_tv,
                                tensor_view_t<5> input_grad_tv,
                                tensor_view_t<5> mean_tv,
                                tensor_view_t<5> mean_grad_tv,
                                tensor_view_t<5> var_grad_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    tensor_layout_t<5> input_grad_layout(input_grad_tv, gid);

    tensor_layout_t<5> layout;
    for(int i = 0; i < 5; i++)
    {
        layout.layout[i] = dims.x[i] ? 0 : input_grad_layout.layout[i];
    }

    FLOAT_ACCUM input_v = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_grad_layout)]);
    FLOAT_ACCUM mean_v  = static_cast<FLOAT_ACCUM>(0);
    if(mean)
    {
        mean_v = CVT_FLOAT2ACCUM(mean[mean_tv.get_tensor_view_idx(layout)]);
    }

    FLOAT_ACCUM input_grad_v = static_cast<FLOAT_ACCUM>(0);
    if(var_grad)
    {
        FLOAT_ACCUM var_grad_v = CVT_FLOAT2ACCUM(var_grad[var_grad_tv.get_tensor_view_idx(layout)]);
        FLOAT_ACCUM res        = var_grad_v * (input_v - mean_v) * static_cast<FLOAT_ACCUM>(2);
        input_grad_v += unbiased ? res / static_cast<FLOAT_ACCUM>(divisor - 1)
                                 : res / static_cast<FLOAT_ACCUM>(divisor);
    }

    if(mean_grad)
    {
        FLOAT_ACCUM mean_grad_v =
            CVT_FLOAT2ACCUM(mean_grad[mean_grad_tv.get_tensor_view_idx(layout)]);
        input_grad_v += mean_grad_v / static_cast<FLOAT_ACCUM>(divisor);
    }

    input_grad[input_grad_tv.get_tensor_view_idx(input_grad_layout)] =
        CVT_ACCUM2FLOAT(input_grad_v);
}

template <typename TIO>
__device__ void VarBackwardContiguousImpl(const TIO* __restrict__ input,
                                          TIO* __restrict__ input_grad,
                                          const TIO* __restrict__ mean,
                                          const TIO* __restrict__ mean_grad,
                                          const TIO* __restrict__ var_grad,
                                          uint64_t N,
                                          dim_5d_t dims,
                                          bool unbiased,
                                          uint32_t divisor,
                                          tensor_view_t<5> input_grad_tv,
                                          tensor_view_t<5> mean_tv,
                                          tensor_view_t<5> mean_grad_tv,
                                          tensor_view_t<5> var_grad_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    tensor_layout_t<5> input_grad_layout(input_grad_tv, gid);

    tensor_layout_t<5> layout;
    for(int i = 0; i < 5; i++)
    {
        layout.layout[i] = dims.x[i] ? 0 : input_grad_layout.layout[i];
    }

    FLOAT_ACCUM input_v = CVT_FLOAT2ACCUM(input[gid]);
    FLOAT_ACCUM mean_v  = static_cast<FLOAT_ACCUM>(0);
    if(mean)
    {
        mean_v = CVT_FLOAT2ACCUM(mean[mean_tv.get_tensor_view_idx(layout)]);
    }

    FLOAT_ACCUM input_grad_v = static_cast<FLOAT_ACCUM>(0);
    if(var_grad)
    {
        FLOAT_ACCUM var_grad_v = CVT_FLOAT2ACCUM(var_grad[var_grad_tv.get_tensor_view_idx(layout)]);
        FLOAT_ACCUM res        = var_grad_v * (input_v - mean_v) * static_cast<FLOAT_ACCUM>(2);
        input_grad_v += unbiased ? res / static_cast<FLOAT_ACCUM>(divisor - 1)
                                 : res / static_cast<FLOAT_ACCUM>(divisor);
    }

    if(mean_grad)
    {
        FLOAT_ACCUM mean_grad_v =
            CVT_FLOAT2ACCUM(mean_grad[mean_grad_tv.get_tensor_view_idx(layout)]);
        input_grad_v += mean_grad_v / static_cast<FLOAT_ACCUM>(divisor);
    }

    input_grad[gid] = CVT_ACCUM2FLOAT(input_grad_v);
}

extern "C" __global__ void VarBackward(const IO_TYPE* input,
                                       IO_TYPE* input_grad,
                                       const IO_TYPE* mean,
                                       const IO_TYPE* mean_grad,
                                       const IO_TYPE* var_grad,
                                       uint64_t N,
                                       dim_5d_t dims,
                                       bool unbiased,
                                       uint32_t divisor,
                                       tensor_view_t<5> input_tv,
                                       tensor_view_t<5> input_grad_tv,
                                       tensor_view_t<5> mean_tv,
                                       tensor_view_t<5> mean_grad_tv,
                                       tensor_view_t<5> var_grad_tv)
{
    VarBackwardImpl<IO_TYPE>(input,
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

extern "C" __global__ void VarBackwardContiguous(const IO_TYPE* __restrict__ input,
                                                 IO_TYPE* __restrict__ input_grad,
                                                 const IO_TYPE* __restrict__ mean,
                                                 const IO_TYPE* __restrict__ mean_grad,
                                                 const IO_TYPE* __restrict__ var_grad,
                                                 uint64_t N,
                                                 dim_5d_t dims,
                                                 bool unbiased,
                                                 uint32_t divisor,
                                                 tensor_view_t<5> input_grad_tv,
                                                 tensor_view_t<5> mean_tv,
                                                 tensor_view_t<5> mean_grad_tv,
                                                 tensor_view_t<5> var_grad_tv)
{
    VarBackwardContiguousImpl<IO_TYPE>(input,
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
