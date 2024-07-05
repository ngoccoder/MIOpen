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
#include <cstddef>
#include <cstdint>

template <class T>
void cpu_diag_forward(tensor<T> input, tensor<T>& ref_output, int64_t diagonal)
{
    if(input.desc.GetLengths().size() == 1)
    {
        auto input_numel = input.desc.GetElementSize();
        auto output_tv   = miopen::get_inner_expanded_tv<2>(ref_output.desc);
        auto offset =
            (diagonal >= 0) ? diagonal * output_tv.stride[1] : -diagonal * output_tv.stride[0];

        par_ford(input_numel)([&](size_t o) {
            long outputIdx        = o * (output_tv.stride[0] + output_tv.stride[1]) + offset;
            ref_output[outputIdx] = input[o];
        });
    }
    else if(input.desc.GetLengths().size() == 2)
    {
        auto output_numel = ref_output.desc.GetElementSize();
        auto input_tv     = miopen::get_inner_expanded_tv<2>(input.desc);
        auto offset =
            (diagonal >= 0) ? diagonal * input_tv.stride[1] : -diagonal * input_tv.stride[0];

        par_ford(output_numel)([&](size_t o) {
            long inputIdx = o * (input_tv.stride[0] + input_tv.stride[1]) + offset;
            ref_output[o] = input[inputIdx];
        });
    }
}

template <class T>
void cpu_diag_backward(tensor<T> outputGrad, tensor<T>& ref_inputGrad, int64_t diagonal)
{
    if(outputGrad.desc.GetLengths().size() == 1)
    {
        auto outputGrad_numel = outputGrad.desc.GetElementSize();
        auto diagonal_tv =
            miopen::solver::diagonal::getDiagonal(ref_inputGrad.desc, diagonal, 0, 1);

        par_ford(outputGrad_numel)([&](size_t o) {
            long inputIdx           = o * diagonal_tv.stride[0] + diagonal_tv.offset;
            ref_inputGrad[inputIdx] = outputGrad[o];
        });
    }
    else if(outputGrad.desc.GetLengths().size() == 2)
    {
        auto outgrad_tv   = miopen::get_inner_expanded_tv<2>(outputGrad.desc);
        auto ingrad_numel = ref_inputGrad.desc.GetElementSize();
        auto offset =
            (diagonal >= 0) ? diagonal * outgrad_tv.stride[1] : -diagonal * outgrad_tv.stride[0];

        par_ford(ingrad_numel)([&](size_t o) {
            long outGradIdx  = o * (outgrad_tv.stride[0] + outgrad_tv.stride[1]) + offset;
            ref_inputGrad[o] = outputGrad[outGradIdx];
        });
    }
}

#endif
