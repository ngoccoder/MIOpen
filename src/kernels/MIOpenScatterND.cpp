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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "hip_atomic.hpp"
#include "tensor_view.hpp"

template <typename TIO, typename TINDEX>
__device__ void ScatterNDAddForward_Kernel(const TIO* input,
                                           const TINDEX* indices,
                                           TIO* output,
                                           size_t num_indices,
                                           size_t slice_size,
                                           size_t ixdim,
                                           tensor_view_t<5> shape)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t i           = 0;
    bool out_of_bounds = false;
    size_t indices_idx = gid / slice_size;
    size_t slice_idx   = gid % slice_size;

    if(indices_idx >= num_indices)
        return;

    size_t output_shape_prefix[5];
    size_t batch_strides[5];

    for(int j = 0; j < ixdim; j++)
    {
        output_shape_prefix[j] = shape.size[j];
    }

    for(int dim = ixdim - 1; dim >= 0; dim--)
    {
        if(dim == ixdim - 1)
        {
            batch_strides[dim] = 1;
        }
        else
        {
            batch_strides[dim] = batch_strides[dim + 1] * output_shape_prefix[dim + 1];
        }
    }

    for(int dim = 0; dim < ixdim; dim++)
    {
        size_t offset = ixdim * indices_idx + dim;
        size_t ix_d   = indices[offset]; // recheck
        out_of_bounds = ix_d >= output_shape_prefix[dim];
        i += ix_d * batch_strides[dim] * slice_size;
    }

    if(!out_of_bounds)
    {
        output[i + slice_idx] = input[gid]; // recheck
    }
}

extern "C" __global__ void ScatterNDAddForward(const IO_TYPE* input,
                                               const INDEX_TYPE* indices,
                                               IO_TYPE* output,
                                               size_t num_indices,
                                               size_t slice_size,
                                               size_t ixdim)
{
    GatherV2BackwardKernel<IO_TYPE, INDEX_TYPE>(outputGrad,
                                                indices,
                                                paramGrad,
                                                outputGrad_tv,
                                                gather_dim_size,
                                                indices_numel,
                                                inner_size,
                                                out_grad_size,
                                                is_axis_zero);
}