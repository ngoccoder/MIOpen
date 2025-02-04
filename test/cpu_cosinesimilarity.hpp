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
#pragma once

#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

#include "tensor_holder.hpp"
#include "tensor_view.hpp"

#include <cstddef>

template <class T>
void cpu_cosinesimilarity_forward(
    const tensor<T>& input1, const tensor<T>& input2, tensor<T>& output, uint32_t dim, float eps)
{
    auto out_sz                   = output.desc.GetElementSize();
    tensor_view_t<5> input1_tv    = miopen::get_inner_expanded_tv<5>(input1.desc);
    tensor_view_t<5> input2_tv    = miopen::get_inner_expanded_tv<5>(input2.desc);
    tensor_view_t<4> output_tv_4d = miopen::get_inner_expanded_tv<4>(output.desc);
    tensor_view_t<5> output_tv    = output_tv_4d.unsqueeze(dim);

    for(auto o = 0; o < out_sz; o++)
    {
        double xy = 0;
        double xn = 0;
        double yn = 0;

        tensor_layout_t<5> out_layout(output_tv, o);

        for(size_t k = 0; k < input1_tv.size[dim]; ++k)
        {
            T x = input1[input1_tv.get_tensor_view_idx(out_layout)];
            T y = input2[input2_tv.get_tensor_view_idx(out_layout)];

            xy += x * y;
            xn += x * x;
            yn += y * y;

            out_layout.layout[dim]++;
        }

        xn = xn > eps ? xn : eps;
        yn = yn > eps ? yn : eps;

        out_layout.layout[dim]                            = 0;
        output[output_tv.get_tensor_view_idx(out_layout)] = xy / sqrt(xn * yn);
    }
}

template <class T>
void cpu_cosinesimilarity_backward(const tensor<T>& input1,
                                   const tensor<T>& input2,
                                   const tensor<T>& output_grad,
                                   tensor<T>& input1_grad,
                                   tensor<T>& input2_grad,
                                   uint32_t dim,
                                   float eps)
{
    auto out_sz                        = output_grad.desc.GetElementSize();
    tensor_view_t<5> input1_tv         = miopen::get_inner_expanded_tv<5>(input1.desc);
    tensor_view_t<5> input2_tv         = miopen::get_inner_expanded_tv<5>(input2.desc);
    tensor_view_t<5> input1_grad_tv    = miopen::get_inner_expanded_tv<5>(input1_grad.desc);
    tensor_view_t<5> input2_grad_tv    = miopen::get_inner_expanded_tv<5>(input2_grad.desc);
    tensor_view_t<4> output_grad_tv_4d = miopen::get_inner_expanded_tv<4>(output_grad.desc);
    tensor_view_t<5> output_grad_tv    = output_grad_tv_4d.unsqueeze(dim);

    for(auto o = 0; o < out_sz; o++)
    {
        tensor_layout_t<5> out_layout(output_grad_tv, o);

        double xy = 0;
        double xn = 0;
        double yn = 0;

        for(size_t k = 0; k < input1_tv.size[dim]; ++k)
        {
            T x = input1[input1_tv.get_tensor_view_idx(out_layout)];
            T y = input2[input2_tv.get_tensor_view_idx(out_layout)];

            xy += x * y;
            xn += x * x;
            yn += y * y;

            out_layout.layout[dim]++;
        }

        xn = xn > eps ? sqrt(xn) : sqrt(eps);
        yn = yn > eps ? sqrt(yn) : sqrt(eps);

        out_layout.layout[dim] = 0;
        T output               = output_grad[output_grad_tv.get_tensor_view_idx(out_layout)];
        double scale           = output / (xn * yn);
        double axpy_scale_x    = -scale * xy / (xn * xn);
        double axpy_scale_y    = -scale * xy / (yn * yn);

        for(size_t k = 0; k < input1_tv.size[dim]; ++k)
        {
            T x = input1[input1_tv.get_tensor_view_idx(out_layout)];
            T y = input2[input2_tv.get_tensor_view_idx(out_layout)];

            input1_grad[input1_grad_tv.get_tensor_view_idx(out_layout)] =
                scale * y + axpy_scale_x * x;
            input2_grad[input2_grad_tv.get_tensor_view_idx(out_layout)] =
                scale * x + axpy_scale_y * y;

            out_layout.layout[dim]++;
        }
    }
}
