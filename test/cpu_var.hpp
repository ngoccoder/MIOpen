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
    auto input_dims = input.desc.GetLengths();
    auto input_grad_dims = input_grad.desc.GetLengths();
    auto mean_dims = mean.desc.GetLengths();
    auto mean_grad_dims = mean_grad.desc.GetLengths();
    auto var_grad_dims = var_grad.desc.GetLengths();

    auto input_grad_numel = std::accumulate(input_grad_dims.begin(), input_grad_dims.end(), 1LL, std::multiplies<int64_t>());

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

        T input_v = input[gid];

        int64_t mean_idx = 0;
        int64_t mean_stride = 1;

        for(int i = mean_dims.size() - 1; i >= 0; --i)
        {
            mean_idx += reduced_idx[i] * mean_stride;
            mean_stride *= mean_dims[i];
        }

        T mean_v = mean[mean_idx];

        T input_grad_v = 0;

        int64_t var_grad_idx = 0;
        int64_t var_grad_stride = 1;

        for(int i = var_grad_dims.size() - 1; i >= 0; --i)
        {
            var_grad_idx += reduced_idx[i] * var_grad_stride;
            var_grad_stride *= var_grad_dims[i];
        }

        if(var_grad.data.size() > 0)
        {
            T var_grad_v = var_grad[var_grad_idx];
            T res = var_grad_v * (input_v - mean_v) * 2;
            input_grad_v += unbiased ? res / (divisor - 1) : res / divisor;
        }

        int64_t mean_grad_idx = 0;
        int64_t mean_grad_stride = 1;

        for(int i = mean_grad_dims.size() - 1; i >= 0; --i)
        {
            mean_grad_idx += reduced_idx[i] * mean_grad_stride;
            mean_grad_stride *= mean_grad_dims[i];
        }

        if(mean_grad.data.size() > 0)
        {
            T mean_grad_v = mean_grad[mean_grad_idx];
            input_grad_v += mean_grad_v / divisor;
        }

        input_grad[gid] = input_grad_v;
    });
}

template <class T>
void cpu_mean(tensor<T> input, tensor<T>& mean, int32_t* dims, int32_t num_dims, int32_t divisor)
{
    auto input_dims = input.desc.GetLengths();
    auto mean_dims = mean.desc.GetLengths();

    auto input_numel = std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>());

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
            if(std::find(dims, dims + num_dims, i) == dims + num_dims)
            {
                reduced_idx[i] = input_idx[i];
            }
        }

        int64_t mean_idx = 0;
        int64_t mean_stride = 1;

        for(int i = mean_dims.size() - 1; i >= 0; --i)
        {
            mean_idx += reduced_idx[i] * mean_stride;
            mean_stride *= mean_dims[i];
        }

        mean[mean_idx] += input[gid];
    });

    auto mean_numel = std::accumulate(mean_dims.begin(), mean_dims.end(), 1LL, std::multiplies<int64_t>());
    for(size_t i = 0; i < mean_numel; ++i)
    {
        mean[i] /= divisor;
    }
}

#endif
