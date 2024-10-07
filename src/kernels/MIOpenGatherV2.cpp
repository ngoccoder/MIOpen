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
#include "tensor_view.hpp"
#include "hip_atomic.hpp"

template <typename TIO, typename TINDEX>
__device__ void GatherV2BackwardKernel(const TIO* outputGrad,
                                       const TINDEX* indices,
                                       TIO* paramGrad,
                                       tensor_view_t<3> outputGrad_tv,
                                       size_t param_grad_numel,
                                       long gather_dim_size,
                                       long indices_size,
                                       long inner_size,
                                       long out_size,
                                       bool is_axis_zero)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_thread = blockDim.x * gridDim.x;

    // Fill zeros
    for (size_t i = gid; i < param_grad_numel; i += num_thread)
    {
        paramGrad[i] = 0;
    }

    if(gid >= out_size)
        return;

    long outer_i   = 0;
    long indices_i = 0;
    long inner_i   = 0;

    if(is_axis_zero)
    {
        indices_i = gid / inner_size;
        inner_i   = gid - indices_i * inner_size;
    }
    else
    {
        long outer_indices_i = gid / inner_size;
        outer_i              = outer_indices_i / indices_size;
        indices_i            = outer_indices_i - outer_i * indices_size;
        inner_i              = gid - outer_indices_i * inner_size;
    }

    size_t gather_i = indices[indices_i];

    if(gather_i < gather_dim_size)
    {
        long param_i = (outer_i * gather_dim_size + gather_i) * inner_size + inner_i;
        FLOAT_ACCUM val =
            CVT_FLOAT2ACCUM(getNDVal(outputGrad, outputGrad_tv, static_cast<uint64_t>(gid)));
        atomic_add_g(paramGrad + param_i, val);
    }
}

extern "C" __global__ void GatherV2Backward(const IO_TYPE* outputGrad,
                                            const INDEX_TYPE* indices,
                                            IO_TYPE* paramGrad,
                                            tensor_view_t<3> outputGrad_tv,
                                            long param_grad_numel,
                                            long gather_dim_size,
                                            long indices_size,
                                            long inner_size,
                                            long out_size,
                                            bool is_axis_zero)
{
    GatherV2BackwardKernel<IO_TYPE, INDEX_TYPE>(outputGrad,
                                                indices,
                                                paramGrad,
                                                outputGrad_tv,
                                                param_grad_numel,
                                                gather_dim_size,
                                                indices_size,
                                                inner_size,
                                                out_size,
                                                is_axis_zero);
}

template <typename TIO, typename TINDEX>
__device__ void BatchedGatherV2BackwardKernel(const TIO* outputGrad,
                                              const TINDEX* indices,
                                              TIO* paramGrad,
                                              tensor_view_t<4> outputGrad_tv,
                                              size_t param_grad_numel,
                                              long outer_size,
                                              long gather_dim_size,
                                              long indices_size,
                                              long slice_size,
                                              long out_size,
                                              bool is_axis_zero,
                                              bool is_batch_dim_zero)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_thread = blockDim.x * gridDim.x;

    // Fill zeros 
    for (size_t i = gid; i < param_grad_numel; i += num_thread)
    {
        paramGrad[i] = 0;
    }


    if(gid >= out_size)
        return;

    long batch_i   = 0;
    long outer_i   = 0;
    long indices_i = 0;
    long inner_i   = 0;

    const long slices_count = gid / slice_size;
    if(is_batch_dim_zero)
    {
        if(is_axis_zero)
        {
            indices_i = slices_count;
        }
        else
        {
            outer_i   = slices_count / indices_size;
            indices_i = slices_count - outer_i * indices_size;
        }
    }
    else
    {
        const long entries_count = slices_count / indices_size;
        if(is_axis_zero)
        {
            batch_i = entries_count;
        }
        else
        {
            batch_i = entries_count / outer_size;
            outer_i = entries_count - batch_i * outer_size;
        }
        indices_i = slices_count - entries_count * slice_size;
    }
    inner_i = gid - slices_count * slice_size;

    size_t gather_i = indices[batch_i * indices_size + indices_i];

    if(gather_i < gather_dim_size)
    {
        long param_i = ((batch_i * outer_size + outer_i) * gather_dim_size) * slice_size + inner_i;
        FLOAT_ACCUM val =
            CVT_FLOAT2ACCUM(getNDVal(outputGrad, outputGrad_tv, static_cast<uint64_t>(gid)));
        atomic_add_g(paramGrad + param_i, val);
    }
}

extern "C" __global__ void BatchedGatherV2Backward(const IO_TYPE* outputGrad,
                                                   const INDEX_TYPE* indices,
                                                   IO_TYPE* paramGrad,
                                                   tensor_view_t<4> outputGrad_tv,
                                                   size_t param_grad_numel,
                                                   long outer_size,
                                                   long gather_dim_size,
                                                   long indices_size,
                                                   long slice_size,
                                                   long out_size,
                                                   bool is_axis_zero,
                                                   bool is_batch_dim_zero)
{
    BatchedGatherV2BackwardKernel<IO_TYPE, INDEX_TYPE>(outputGrad,
                                                       indices,
                                                       paramGrad,
                                                       outputGrad_tv,
                                                       param_grad_numel,
                                                       outer_size,
                                                       gather_dim_size,
                                                       indices_size,
                                                       slice_size,
                                                       out_size,
                                                       is_axis_zero,
                                                       is_batch_dim_zero);
}
