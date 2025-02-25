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

#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include <miopen/tensor_view_utils.hpp>
#include <vector>

template <class T>
void cpu_var_backward(const tensor<T>& input,
                      tensor<T>& input_grad,
                      const tensor<T>& mean,
                      const tensor<T>& mean_grad,
                      const tensor<T>& var_grad,
                      dim_5d_t dims,
                      bool unbiased)
{
    tensor_view_t<5> input_tv      = miopen::get_inner_expanded_tv<5>(input.desc);
    tensor_view_t<5> mean_tv       = miopen::get_inner_expanded_tv<5>(mean.desc);
    tensor_view_t<5> mean_grad_tv  = miopen::get_inner_expanded_tv<5>(mean_grad.desc);
    tensor_view_t<5> var_grad_tv   = miopen::get_inner_expanded_tv<5>(var_grad.desc);
    tensor_view_t<5> input_grad_tv = miopen::get_inner_expanded_tv<5>(input_grad.desc);

    uint32_t divisor = 1;
    auto input_len   = input.desc.GetLengths();
    for(auto i = 0; i < 5; i++)
    {
        divisor *= (dims.x[i] ? input_len[i] : 1);
    }

    auto input_grad_numel = input_grad.desc.GetElementSize();
    std::fill(input_grad.data.begin(), input_grad.data.end(), 0);

    par_ford(input_grad_numel)([&](size_t gid) {
        tensor_layout_t<5> input_grad_layout(input_grad_tv, gid);

        tensor_layout_t<5> layout;
        for(int i = 0; i < 5; i++)
        {
            layout.layout[i] = dims.x[i] ? 0 : input_grad_layout.layout[i];
        }

        double input_v =
            static_cast<double>(input[input_tv.get_tensor_view_idx(input_grad_layout)]);
        double mean_v = static_cast<double>(0);

        if(mean.data.data())
        {
            mean_v = static_cast<double>(mean[mean_tv.get_tensor_view_idx(layout)]);
        }

        double input_grad_v = static_cast<double>(0.0);
        if(var_grad.data.data())
        {
            double var_grad_v =
                static_cast<double>(var_grad[var_grad_tv.get_tensor_view_idx(layout)]);
            double res = var_grad_v * (input_v - mean_v) * static_cast<double>(2);
            input_grad_v += unbiased ? res / static_cast<double>(divisor - 1)
                                     : res / static_cast<double>(divisor);
        }

        if(mean_grad.data.data())
        {
            double mean_grad_v =
                static_cast<double>(mean_grad[mean_grad_tv.get_tensor_view_idx(layout)]);
            input_grad_v += mean_grad_v / static_cast<double>(divisor);
        }

        input_grad[input_grad_tv.get_tensor_view_idx(input_grad_layout)] =
            static_cast<T>(input_grad_v);
    });
}

template <class T>
void cpu_mean(const tensor<T>& input, tensor<T>& mean, const std::vector<int>& dims_vector)
{
    uint32_t divisor = 1;
    auto input_len   = input.desc.GetLengths();
    for(auto dim : dims_vector)
    {
        divisor *= input_len[dim];
    }

    auto input_tv = miopen::get_inner_expanded_tv<5>(input.desc);
    auto mean_tv  = miopen::get_inner_expanded_tv<5>(mean.desc);

    auto input_numel = input.desc.GetElementSize();

    std::fill(mean.data.begin(), mean.data.end(), 0);

    par_ford(input_numel)([&](size_t gid) {
        tensor_layout_t<5> input_layout(input_tv, gid);
        tensor_layout_t<5> mean_layout = input_layout;

        for(auto dim : dims_vector)
        {
            mean_layout.layout[dim] = 0;
        }

        mean[mean_tv.get_tensor_view_idx(mean_layout)] +=
            input[input_tv.get_tensor_view_idx(input_layout)];
    });

    auto mean_numel = mean.desc.GetElementSize();
    for(size_t i = 0; i < mean_numel; ++i)
    {
        mean[i] /= static_cast<T>(divisor);
    }
}
