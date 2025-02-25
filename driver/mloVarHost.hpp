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

#include "tensor_view.hpp"
#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloVarBackwardRunHost(const miopenTensorDescriptor_t inputDesc,
                              const miopenTensorDescriptor_t inputGradDesc,
                              const miopenTensorDescriptor_t meanDesc,
                              const miopenTensorDescriptor_t meanGradDesc,
                              const miopenTensorDescriptor_t varGradDesc,
                              const Tgpu* input,
                              Tcheck* input_grad,
                              const Tgpu* mean,
                              const Tgpu* mean_grad,
                              const Tgpu* var_grad,
                              dim_5d_t dims,
                              bool unbiased)
{
    tensor_view_t<5> input_tv      = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    tensor_view_t<5> input_grad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    tensor_view_t<5> mean_tv       = miopen::get_inner_expanded_tv<5>(miopen::deref(meanDesc));
    tensor_view_t<5> mean_grad_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(meanGradDesc));
    tensor_view_t<5> var_grad_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(varGradDesc));

    uint32_t divisor = 1;
    auto input_len   = miopen::deref(inputDesc).GetLengths();
    for(auto i = 0; i < 5; i++)
    {
        divisor *= (dims.x[i] ? input_len[i] : 1);
    }

    auto input_grad_numel = miopen::deref(inputGradDesc).GetElementSize();
    std::fill(input_grad, input_grad + input_grad_numel, 0);

    for(size_t gid = 0; gid < input_grad_numel; ++gid)
    {
        tensor_layout_t<5> input_grad_layout(input_grad_tv, gid);

        tensor_layout_t<5> layout;
        for(int i = 0; i < 5; i++)
        {
            layout.layout[i] = dims.x[i] ? 0 : input_grad_layout.layout[i];
        }

        double input_v =
            static_cast<double>(input[input_tv.get_tensor_view_idx(input_grad_layout)]);
        double mean_v = static_cast<double>(0);

        if(mean)
        {
            mean_v = static_cast<double>(mean[mean_tv.get_tensor_view_idx(layout)]);
        }

        double input_grad_v = static_cast<double>(0);
        if(var_grad)
        {
            double var_grad_v =
                static_cast<double>(var_grad[var_grad_tv.get_tensor_view_idx(layout)]);
            double res = var_grad_v * (input_v - mean_v) * static_cast<double>(2);
            input_grad_v += unbiased ? res / static_cast<double>(divisor - 1)
                                     : res / static_cast<double>(divisor);
        }

        if(mean_grad)
        {
            double mean_grad_v =
                static_cast<double>(mean_grad[mean_grad_tv.get_tensor_view_idx(layout)]);
            input_grad_v += mean_grad_v / static_cast<double>(divisor);
        }

        input_grad[input_grad_tv.get_tensor_view_idx(input_grad_layout)] =
            static_cast<Tcheck>(input_grad_v);
    }

    return 0;
}

template <typename Tgpu>
int32_t mloMeanForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                              const miopenTensorDescriptor_t meanDesc,
                              const Tgpu* input,
                              Tgpu* mean,
                              const std::vector<int>& dims)
{
    uint32_t divisor = 1;
    auto input_len   = miopen::deref(inputDesc).GetLengths();
    for(auto dim : dims)
    {
        divisor *= input_len[dim];
    }

    auto input_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto mean_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(meanDesc));

    auto input_numel = miopen::deref(inputDesc).GetElementSize();

    for(size_t gid = 0; gid < input_numel; ++gid)
    {
        tensor_layout_t<5> input_layout(input_tv, gid);
        tensor_layout_t<5> mean_layout = input_layout;
        for(auto dim : dims)
        {
            mean_layout.layout[dim] = 0;
        }

        mean[mean_tv.get_tensor_view_idx(mean_layout)] +=
            input[input_tv.get_tensor_view_idx(input_layout)];
    }

    auto mean_numel = miopen::deref(meanDesc).GetElementSize();
    for(size_t i = 0; i < mean_numel; ++i)
    {
        mean[i] /= static_cast<Tgpu>(divisor);
    }

    return 0;
}
