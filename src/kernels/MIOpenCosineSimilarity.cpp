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

template <typename TIO>
__device__ void CosineSimilarityForwardKernel(const TIO* input1,
                                              const TIO* input2,
                                              TIO* output,
                                              uint64_t output_size,
                                              uint32_t dim,
                                              float eps,
                                              tensor_view_t<5> input1_tv,
                                              tensor_view_t<5> input2_tv,
                                              tensor_view_t<5> output_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= output_size)
        return;

    tensor_layout_t<5> out_layout(output_tv, gid);

    FLOAT_ACCUM xy = 0;
    FLOAT_ACCUM xn = 0;
    FLOAT_ACCUM yn = 0;

    for(size_t k = 0; k < input1_tv.size[dim]; ++k)
    {
        TIO x = input1[input1_tv.get_tensor_view_idx(out_layout)];
        TIO y = input2[input2_tv.get_tensor_view_idx(out_layout)];

        xy += CVT_FLOAT2ACCUM(x) * CVT_FLOAT2ACCUM(y);
        xn += CVT_FLOAT2ACCUM(x) * CVT_FLOAT2ACCUM(x);
        yn += CVT_FLOAT2ACCUM(y) * CVT_FLOAT2ACCUM(y);

        out_layout.layout[dim]++;
    }

    xn = xn > eps ? xn : eps;
    yn = yn > eps ? yn : eps;

    out_layout.layout[dim]                            = 0;
    output[output_tv.get_tensor_view_idx(out_layout)] = CVT_ACCUM2FLOAT(xy / sqrtf(xn * yn));
}

extern "C" __global__ void CosineSimilarityForward(const IO_TYPE* input1,
                                                   const IO_TYPE* input2,
                                                   IO_TYPE* output,
                                                   uint64_t output_size,
                                                   uint32_t dim,
                                                   float eps,
                                                   tensor_view_t<5> input1_tv,
                                                   tensor_view_t<5> input2_tv,
                                                   tensor_view_t<5> output_tv)
{
    CosineSimilarityForwardKernel<IO_TYPE>(
        input1, input2, output, output_size, dim, eps, input1_tv, input2_tv, output_tv);
}

template <typename TIO>
__device__ void CosineSimilarityBackwardKernel(const TIO* input1,
                                               const TIO* input2,
                                               const TIO* output_grad,
                                               TIO* input1_grad,
                                               TIO* input2_grad,
                                               uint64_t output_size,
                                               uint32_t dim,
                                               float eps,
                                               tensor_view_t<5> input1_tv,
                                               tensor_view_t<5> input2_tv,
                                               tensor_view_t<5> output_grad_tv,
                                               tensor_view_t<5> input1_grad_tv,
                                               tensor_view_t<5> input2_grad_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= output_size)
        return;

    tensor_layout_t<5> out_layout(output_grad_tv, gid);

    FLOAT_ACCUM xy = 0;
    FLOAT_ACCUM xn = 0;
    FLOAT_ACCUM yn = 0;

    for(size_t k = 0; k < input1_tv.size[dim]; ++k)
    {
        TIO x = input1[input1_tv.get_tensor_view_idx(out_layout)];
        TIO y = input2[input2_tv.get_tensor_view_idx(out_layout)];

        xy += CVT_FLOAT2ACCUM(x) * CVT_FLOAT2ACCUM(y);
        xn += CVT_FLOAT2ACCUM(x) * CVT_FLOAT2ACCUM(x);
        yn += CVT_FLOAT2ACCUM(y) * CVT_FLOAT2ACCUM(y);

        out_layout.layout[dim]++;
    }

    xn = xn > eps ? sqrt(xn) : sqrt(eps);
    yn = yn > eps ? sqrt(yn) : sqrt(eps);

    out_layout.layout[dim]   = 0;
    TIO output               = output_grad[output_grad_tv.get_tensor_view_idx(out_layout)];
    FLOAT_ACCUM scale        = CVT_FLOAT2ACCUM(output) / (xn * yn);
    FLOAT_ACCUM axpy_scale_x = -scale * xy / (xn * xn);
    FLOAT_ACCUM axpy_scale_y = -scale * xy / (yn * yn);

    for(size_t k = 0; k < input1_tv.size[dim]; ++k)
    {
        FLOAT_ACCUM x = CVT_ACCUM2FLOAT(input1[input1_tv.get_tensor_view_idx(out_layout)]);
        FLOAT_ACCUM y = CVT_ACCUM2FLOAT(input2[input2_tv.get_tensor_view_idx(out_layout)]);

        if(input1_grad)
        {
            input1_grad[input1_grad_tv.get_tensor_view_idx(out_layout)] =
                scale * y + axpy_scale_x * x;
        }
        if(input2_grad)
        {
            input2_grad[input2_grad_tv.get_tensor_view_idx(out_layout)] =
                scale * x + axpy_scale_y * y;
        }

        out_layout.layout[dim]++;
    }
}

extern "C" __global__ void CosineSimilarityBackward(const IO_TYPE* input1,
                                                    const IO_TYPE* input2,
                                                    const IO_TYPE* output_grad,
                                                    IO_TYPE* input1_grad,
                                                    IO_TYPE* input2_grad,
                                                    uint64_t output_size,
                                                    uint32_t dim,
                                                    float eps,
                                                    tensor_view_t<5> input1_tv,
                                                    tensor_view_t<5> input2_tv,
                                                    tensor_view_t<5> output_grad_tv,
                                                    tensor_view_t<5> input1_grad_tv,
                                                    tensor_view_t<5> input2_grad_tv)
{
    CosineSimilarityBackwardKernel<IO_TYPE>(input1,
                                            input2,
                                            output_grad,
                                            input1_grad,
                                            input2_grad,
                                            output_size,
                                            dim,
                                            eps,
                                            input1_tv,
                                            input2_tv,
                                            output_grad_tv,
                                            input1_grad_tv,
                                            input2_grad_tv);
}
