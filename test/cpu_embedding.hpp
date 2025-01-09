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

#include "tensor_holder.hpp"
#include "tensor_view.hpp"

#include <cstddef>

template <class T>
void cpu_embedding_backward(const tensor<int64_t>& input,
                            const tensor<T>& output_grad,
                            tensor<T>& ref_weight_grad,
                            std::vector<int32_t>& indices_freq,
                            int64_t padding_idx)
{
    auto input_tv       = miopen::get_inner_expanded_tv<4>(input.desc);
    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(output_grad.desc);
    auto weight_grad_tv = miopen::get_inner_expanded_tv<2>(ref_weight_grad.desc);
    auto embedding_dim  = weight_grad_tv.size[1];
    auto num_embeddings = weight_grad_tv.size[0];
    auto out_grad_numel = output_grad.desc.GetElementSize();

    for(size_t o = 0; o < out_grad_numel; o++)
    {
        size_t i = o / embedding_dim, j = o % embedding_dim;

        tensor_layout_t<4> input_layout(input_tv, i);
        size_t input_idx      = input_tv.get_tensor_view_idx(input_layout);
        int64_t embedding_idx = input[input_idx];

        if(embedding_idx == padding_idx)
            continue;

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            T scale                = indices_freq.data()
                                         ? static_cast<T>(1.0f) / static_cast<T>(indices_freq[input_idx])
                                         : static_cast<T>(1.0f);
            size_t weight_grad_idx = weight_grad_tv.get_tensor_view_idx({embedding_idx, j});
            tensor_layout_t<5> outGrad_layout(output_grad_tv, o);
            ref_weight_grad[weight_grad_idx] +=
                output_grad[output_grad_tv.get_tensor_view_idx(outGrad_layout)] * scale;
        }
    }
}
