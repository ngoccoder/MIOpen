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
#include "miopen/diagonal/solvers.hpp"
#include "miopen/tensor_layout.hpp"
#include "miopen/tensor_view_utils.hpp"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include <cstddef>
#include <cstdint>

template <class T>
void cpu_diagembed_forward(
    tensor<T> input, tensor<T>& ref_output, int64_t offset, int64_t dim1, int64_t dim2)
{
    auto diag_tv     = miopen::solver::diagonal::getDiagonal(ref_output.desc, offset, dim1, dim2);
    auto input_tv    = miopen::get_inner_expanded_tv<5>(input.desc);
    auto input_numel = input.desc.GetElementSize();

    for(size_t i = 0; i < input_numel; i++)
    {
        auto layout   = tensor_layout_t<5>(input_tv, i);
        auto inputIdx = input_tv.get_tensor_view_idx(layout);

        auto outLayout = tensor_layout_t<5>(diag_tv, i);
        auto outIdx    = diag_tv.get_tensor_view_idx(outLayout);

        ref_output[outIdx] = input[inputIdx];
    }
}

#endif
