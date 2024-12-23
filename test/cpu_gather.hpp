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

#include <miopen/tensor_view_utils.hpp>
#include "tensor_holder.hpp"

template <class T, class TID>
void cpu_gathernd_backward(const tensor<T>& outputGrad,
                           const tensor<TID>& indices,
                           tensor<T>& paramGrad)
{
    auto indices_num_dim    = indices.desc.GetNumDims();
    auto indices_len        = indices.desc.GetLengths();
    auto param_grad_num_dim = paramGrad.desc.GetNumDims();
    auto param_grad_len     = paramGrad.desc.GetLengths();
    size_t slice_dim        = (indices_num_dim > 1) ? indices_len[indices_num_dim - 1] : 1;

    size_t slice_size = 1;
    for(size_t i = slice_dim; i < param_grad_num_dim; i++)
    {
        slice_size *= param_grad_len[i];
    }

    size_t batch_strides[9];

    for(int dim = slice_dim - 1; dim >= 0; dim--)
    {
        if(static_cast<size_t>(dim) == slice_dim - 1)
        {
            batch_strides[dim] = 1;
        }
        else
        {
            batch_strides[dim] = batch_strides[dim + 1] * param_grad_len[dim + 1];
        }
    }

    size_t output_grad_size = outputGrad.desc.GetElementSize();
    for(size_t i = 0; i < output_grad_size; i++)
    {
        bool out_of_bounds = false;
        size_t indices_idx = i / slice_size;
        size_t slice_idx   = i % slice_size;

        size_t param_grad_idx = 0;
        // Get 1D tensor (length = d) out of indices and get index into param_grad
        for(size_t dim = 0; dim < slice_dim; dim++)
        {
            size_t offset = slice_dim * indices_idx + dim;
            size_t ix_d   = indices[offset];
            out_of_bounds |= ix_d >= param_grad_len[dim];
            param_grad_idx += ix_d * batch_strides[dim] * slice_size;
        }

        if(!out_of_bounds)
        {
            paramGrad[param_grad_idx + slice_idx] += outputGrad[i];
        }
    }
}
