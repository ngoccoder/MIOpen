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

#ifndef GUARD_CPU_VAR_HPP
#define GUARD_CPU_VAR_HPP

#include "tensor_holder.hpp"
#include <atomic>
#include <vector>

template <class T>
void cpu_var_backward(tensor<T> input,
                      tensor<T>& input_grad,
                      tensor<T> mean,
                      tensor<T> mean_grad,
                      tensor<T> var_grad,
                      int32_t* dims,
                      int32_t num_dims,
                      bool unbiased,
                      int32_t divisor)
{
    auto input_dims      = input.desc.GetLengths();
    auto input_strides   = input.desc.GetStrides();
    auto input_grad_dims = input_grad.desc.GetLengths();
    auto input_grad_strides = input_grad.desc.GetStrides();
    auto mean_dims       = mean.desc.GetLengths();
    auto mean_strides    = mean.desc.GetStrides();
    auto mean_grad_dims  = mean_grad.desc.GetLengths();
    auto mean_grad_strides = mean_grad.desc.GetStrides();
    auto var_grad_dims   = var_grad.desc.GetLengths();
    auto var_grad_strides = var_grad.desc.GetStrides();

    auto input_grad_numel = std::accumulate(
        input_grad_dims.begin(), input_grad_dims.end(), 1LL, std::multiplies<int64_t>());

    std::fill(input_grad.data.begin(), input_grad.data.end(), 0);

    par_ford(input_grad_numel)([&](size_t gid) {
        std::vector<int64_t> input_idx(input_dims.size(), 0);
        int64_t tmp_gid = gid;

        for(int i = input_dims.size() - 1; i >= 0; --i)
        {
            input_idx[i] = tmp_gid % input_dims[i];
            tmp_gid /= input_dims[i];
        }

        std::vector<int64_t> reduced_idx(input_dims.size(), 0);
        for(int i = 0; i < input_dims.size(); ++i)
        {
            if(std::find(dims, dims + num_dims, i) == dims + num_dims)
            {
                reduced_idx[i] = input_idx[i];
            }
        }

        T input_v = input[std::inner_product(input_idx.begin(), input_idx.end(), input_strides.begin(), static_cast<size_t>(0))];

        int64_t mean_idx = std::inner_product(reduced_idx.begin(), reduced_idx.end(), mean_strides.begin(), static_cast<size_t>(0));
        T mean_v = mean[mean_idx];

        T input_grad_v = static_cast<T>(0.0);

        int64_t var_grad_idx = std::inner_product(reduced_idx.begin(), reduced_idx.end(), var_grad_strides.begin(), static_cast<size_t>(0));
        T var_grad_v = var_grad[var_grad_idx];
        T res = static_cast<T>(var_grad_v * (static_cast<float>(input_v) - static_cast<float>(mean_v)) * 2.0f);
        input_grad_v += unbiased ? res / static_cast<T>(divisor - 1) : res / static_cast<T>(divisor);

        int64_t mean_grad_idx = std::inner_product(reduced_idx.begin(), reduced_idx.end(), mean_grad_strides.begin(), static_cast<size_t>(0));
        T mean_grad_v = mean_grad[mean_grad_idx];
        input_grad_v += mean_grad_v / static_cast<T>(divisor);

        input_grad[std::inner_product(input_idx.begin(), input_idx.end(), input_grad_strides.begin(), static_cast<size_t>(0))] = input_grad_v;
    });
}

template <class T>
void cpu_mean(tensor<T> input, tensor<T>& mean, std::vector<int32_t>& dims_vector, int32_t divisor)
{
    auto input_dims = input.desc.GetLengths();
    auto input_strides = input.desc.GetStrides();
    auto mean_dims  = mean.desc.GetLengths();
    auto mean_strides = mean.desc.GetStrides();

    auto input_numel =
        std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>());

    par_ford(input_numel)([&](size_t gid) {
        std::vector<int64_t> input_idx(input_dims.size(), 0);
        int64_t tmp_gid = gid;

        for(int i = input_dims.size() - 1; i >= 0; --i)
        {
            input_idx[i] = tmp_gid % input_dims[i];
            tmp_gid /= input_dims[i];
        }

        std::vector<int64_t> reduced_idx(input_dims.size(), 0);
        for(int i = 0; i < input_dims.size(); ++i)
        {
            if(std::find(dims_vector.begin(), dims_vector.end(), i) == dims_vector.end())
            {
                reduced_idx[i] = input_idx[i];
            }
        }

        int64_t mean_idx = std::inner_product(reduced_idx.begin(), reduced_idx.end(), mean_strides.begin(), static_cast<size_t>(0));
        mean[mean_idx] += input[std::inner_product(input_idx.begin(), input_idx.end(), input_strides.begin(), static_cast<size_t>(0))];
    });

    auto mean_numel =
        std::accumulate(mean_dims.begin(), mean_dims.end(), 1LL, std::multiplies<int64_t>());
    for(size_t i = 0; i < mean_numel; ++i)
    {
        mean[i] /= static_cast<T>(divisor);
    }
}


#endif
