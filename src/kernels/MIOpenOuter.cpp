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
__device__ void
OuterForwardImpl(const TIO* x1, const TIO* x2, TIO* y, size_t y_numel, tensor_view_t<2> y_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= y_numel)
        return;

    tensor_layout_t<2> y_layout(y_tv, gid);
    y[y_tv.get_tensor_view_idx(y_layout)] = x1[y_layout.layout[0]] * x2[y_layout.layout[1]];
}

extern "C" __global__ void OuterForward(
    const IO_TYPE* x1, const IO_TYPE* x2, IO_TYPE* y, size_t y_numel, tensor_view_t<2> y_tv)
{
    OuterForwardImpl<IO_TYPE>(x1, x2, y, y_numel, y_tv);
}

template <typename TIO>
__device__ void OuterBackwardImpl(const TIO* x1,
                                  const TIO* x2,
                                  const TIO* y_grad,
                                  TIO* x1_grad,
                                  TIO* x2_grad,
                                  tensor_view_t<2> y_grad_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    auto x1_numel = y_grad_tv.size[0];
    auto x2_numel = y_grad_tv.size[1];

    if(gid >= max(x1_numel, x2_numel))
        return;

    if(x1_grad && gid < x1_numel)
    {
        TIO sum = 0;
        for(auto i = 0; i < x2_numel; ++i)
        {
            sum += x2[i] * y_grad[y_grad_tv.get_tensor_view_idx({gid, i})];
        }
        x1_grad[gid] = sum;
    }

    if(x2_grad && gid < x2_numel)
    {
        TIO sum = 0;
        for(auto i = 0; i < x1_numel; ++i)
        {
            sum += x1[i] * y_grad[y_grad_tv.get_tensor_view_idx({i, gid})];
        }
        x2_grad[gid] = sum;
    }
}

extern "C" __global__ void OuterBackward(const IO_TYPE* x1,
                                         const IO_TYPE* x2,
                                         const IO_TYPE* y_grad,
                                         IO_TYPE* x1_grad,
                                         IO_TYPE* x2_grad,
                                         tensor_view_t<2> y_grad_tv)
{
    OuterBackwardImpl<IO_TYPE>(x1, x2, y_grad, x1_grad, x2_grad, y_grad_tv);
}
