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

#include <cstddef>
#include <cstdint>

#include <miopen/tensor_view_utils.hpp>
#include "tensor_holder.hpp"
#include "tensor_view.hpp"

template <class T>
void cpu_gather_forward(const tensor<T>& input,
                        const tensor<int64_t>& indices,
                        tensor<T>& output,
                        uint32_t dim)
{
    auto output_size = output.desc.GetElementSize();
    auto input_tv    = miopen::get_inner_expanded_tv<5>(input.desc);
    auto indices_tv  = miopen::get_inner_expanded_tv<5>(indices.desc);
    auto output_tv   = miopen::get_inner_expanded_tv<5>(output.desc);
    for(size_t i = 0; i < output_size; i++)
    {
        tensor_layout_t<5> output_layout(output_tv, i);
        if(output_layout.layout[0] >= output_tv.size[0]) // out of bound
            continue;

        size_t output_idx         = output_tv.get_tensor_view_idx(output_layout);
        output_layout.layout[dim] = indices[indices_tv.get_tensor_view_idx(output_layout)];
        if(output_layout.layout[dim] >= input_tv.size[dim]) // out of bound
            continue;

        size_t input_idx   = input_tv.get_tensor_view_idx(output_layout);
        output[output_idx] = input[input_idx];
    };
}
