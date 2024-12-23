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
#pragma once

#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

#include "ford.hpp"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"

#include <cstddef>

template <class T>
void cpu_trace_forward(const tensor<T>& input, tensor<T>& ref_output)
{
    tensor_view_t<2> input_tv = miopen::get_inner_expanded_tv<2>(input.desc);
    auto input_len            = input.desc.GetLengths();
    size_t N                  = std::min(input_len[0], input_len[1]);
    double res                = 0;

    for(size_t i = 0; i < N; i++)
    {
        tensor_layout_t<2> input_layout = {i, i};
        size_t input_idx                = input_tv.get_tensor_view_idx(input_layout);
        T val                           = input[input_idx];
        res += static_cast<double>(input[input_idx]);
    }
    ref_output[0] = static_cast<T>(res);
}

template <class T>
void cpu_trace_backward(const tensor<T>& output_grad, tensor<T>& ref_input_grad)
{
    tensor_view_t<2> input_grad_tv = miopen::get_inner_expanded_tv<2>(ref_input_grad.desc);
    auto input_grad_len            = ref_input_grad.desc.GetLengths();
    size_t N                       = input_grad_len[0];

    par_ford(N)([&](size_t i) {
        size_t idx = i % (input_grad_tv.size[1] + 1);

        if(idx != input_grad_tv.size[1])
        {
            T val                                                            = output_grad[0];
            tensor_layout_t<2> ingrad_layout                                 = {i, idx};
            ref_input_grad[input_grad_tv.get_tensor_view_idx(ingrad_layout)] = static_cast<T>(val);
        }
    });
}
