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
#ifndef GUARD_CPU_DIAG_HPP
#define GUARD_CPU_DIAG_HPP

#include "ford.hpp"
#include "miopen/tensor_layout.hpp"
#include "miopen/tensor_view_utils.hpp"
#include "tensor_holder.hpp"
#include <cstddef>
#include <cstdint>

template <class T>
void cpu_diagflat_forward(tensor<T> input, tensor<T>& ref_output, int64_t offset)
{
    auto output_tv       = miopen::get_inner_expanded_tv<2>(ref_output.desc);
    auto output_stride_0 = output_tv.stride[0];
    auto output_stride_1 = output_tv.stride[1];
    auto output_offset   = (offset >= 0) ? offset * output_stride_1 : -offset * output_stride_0;
    auto input_numel     = input.desc.GetElementSize();
    for(size_t i = 0; i < input_numel; i++)
    {
        long outputIdx        = i * (output_stride_0 + output_stride_1) + output_offset;
        ref_output[outputIdx] = input[i];
    }
}

#endif
