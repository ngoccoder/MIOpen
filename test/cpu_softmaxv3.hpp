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

#include <cstdint>
#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

#include "tensor_holder.hpp"

#include <cstddef>

template <class T>
void cpu_softmax_contiguous_forward(const tensor<T>& input,
                                    tensor<T>& ref_output,
                                    uint32_t dim,
                                    miopenSoftmaxAlgorithm_t algorithm)
{
    auto input_len   = input.desc.GetLengths();
    auto reduce_size = input_len[dim];

    uint64_t inner_size = 1;
    for(size_t i = dim + 1; i < input_len.size(); i++)
    {
        inner_size *= input_len[i];
    }

    uint64_t outer_size = 1;
    for(uint32_t i = 0; i < dim; i++)
    {
        outer_size *= input_len[i];
    }

    for(uint64_t o = 0; o < outer_size; o++)
    {
        for(uint64_t i = 0; i < inner_size; i++)
        {
            T pmax          = std::numeric_limits<T>::min();
            size_t base_idx = o * reduce_size * inner_size + i;
            for(uint64_t r = 0; r < reduce_size; r++)
            {
                pmax = std::max(pmax, input[base_idx + r * inner_size]);
            }

            double psum = 0;
            for(uint64_t r = 0; r < reduce_size; r++)
            {
                double val = exp(static_cast<double>(input[base_idx + r * inner_size]) -
                                 static_cast<double>(pmax));
                psum += val;
            }

            for(uint64_t r = 0; r < reduce_size; r++)
            {
                ref_output[base_idx + r * inner_size] =
                    (algorithm == MIOPEN_SOFTMAX_LOG)
                        ? input[base_idx + r * inner_size] - pmax - log(psum)
                        : exp(input[base_idx + r * inner_size] - pmax) / psum;
            }
        }
    }
}

template <class T>
void cpu_softmax_contiguous_backward(const tensor<T>& output,
                                     const tensor<T>& output_grad,
                                     tensor<T>& ref_input_grad,
                                     uint32_t dim,
                                     miopenSoftmaxAlgorithm_t algorithm)
{
    auto output_len     = output.desc.GetLengths();
    auto reduce_size    = output_len[dim];
    uint64_t inner_size = 1;
    for(size_t i = dim + 1; i < output_len.size(); i++)
    {
        inner_size *= output_len[i];
    }

    uint64_t outer_size = 1;
    for(uint32_t i = 0; i < dim; i++)
    {
        outer_size *= output_len[i];
    }

    for(uint64_t o = 0; o < outer_size; o++)
    {
        for(uint64_t i = 0; i < inner_size; i++)
        {
            double psum     = 0;
            size_t base_idx = o * reduce_size * inner_size + i;
            for(uint64_t r = 0; r < reduce_size; r++)
            {
                if(algorithm == MIOPEN_SOFTMAX_LOG)
                {
                    psum += output_grad[base_idx + r * inner_size];
                }
                else
                {
                    psum +=
                        output_grad[base_idx + r * inner_size] * output[base_idx + r * inner_size];
                }
            }

            for(uint64_t r = 0; r < reduce_size; r++)
            {
                if(algorithm == MIOPEN_SOFTMAX_LOG)
                {
                    ref_input_grad[base_idx + r * inner_size] =
                        output_grad[base_idx + r * inner_size] -
                        psum * exp(output[base_idx + r * inner_size]);
                }
                else
                {
                    ref_input_grad[base_idx + r * inner_size] =
                        (output_grad[base_idx + r * inner_size] - psum) *
                        output[base_idx + r * inner_size];
                }
            }
        }
    }
}
