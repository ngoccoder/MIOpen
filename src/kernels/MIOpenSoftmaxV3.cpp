/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <limits>

#define LOCAL_SIZE 256
#define CHUNK_SIZE 64

template <typename TIO>
__device__ void SoftmaxAccurateFwdDimIsNotLastContiguousKernel(
    const TIO* input, TIO* output, uint64_t reduce_size, uint64_t inner_size, int32_t algorithm)
{
    size_t gid = blockIdx.x;
    size_t tid = threadIdx.x;

    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    uint64_t innerspace_width  = inner_size;
    uint64_t innerspace_height = reduce_size;

    uint64_t chunk_width  = CHUNK_SIZE;
    uint64_t chunk_height = LOCAL_SIZE / chunk_width;

    uint64_t num_chunks_in_innerspace_row = (innerspace_width + chunk_width - 1) / chunk_width;
    uint64_t gid_in_innerspace_row        = (gid % num_chunks_in_innerspace_row) * chunk_width;
    uint64_t innerspace_base_index =
        (gid / num_chunks_in_innerspace_row) * innerspace_height * innerspace_width;

    uint64_t chunk_x = tid % chunk_width;
    uint64_t chunk_y = tid / chunk_width;

    uint64_t inner_x = gid_in_innerspace_row + chunk_x;

    // MAX REDUCTION
    FLOAT_ACCUM pmax = std::numeric_limits<FLOAT_ACCUM>::min();

    if(inner_x < innerspace_width)
    {
        for(uint64_t i = 0; i < innerspace_height; i += chunk_height)
        {
            uint64_t inner_y = i + chunk_y;
            if(inner_y >= innerspace_height)
            {
                break;
            }
            uint64_t idx = inner_y * innerspace_width + innerspace_base_index;

            pmax = max(pmax, CVT_FLOAT2ACCUM(input[idx]));
        }
    }

    ltmp[tid] = pmax;
    __syncthreads();
    for(uint64_t i = LOCAL_SIZE >> 1; i >= chunk_width; i >>= 1)
    {
        if(tid < i)
        {
            ltmp[tid] = max(ltmp[tid], ltmp[tid + i]);
        }
        __syncthreads();
    }
    pmax = ltmp[tid % chunk_width];

    // SUM REDUCTION
    FLOAT_ACCUM psum = 0;

    if(inner_x < innerspace_width)
    {
        for(uint64_t i = 0; i < innerspace_height; i += chunk_height)
        {
            uint64_t inner_y = i + chunk_y;
            if(inner_y >= innerspace_height)
            {
                break;
            }
            uint64_t idx = inner_y * innerspace_width + inner_x + innerspace_base_index;

            psum += exp(CVT_FLOAT2ACCUM(input[idx]) - pmax);
        }
    }

    ltmp[tid] = psum;
    __syncthreads();
    for(uint64_t i = LOCAL_SIZE >> 1; i >= chunk_width; i >>= 1)
    {
        if(tid < i)
        {
            ltmp[tid] += ltmp[tid + i];
        }
        __syncthreads();
    }

    // Final normalization
    psum = ltmp[tid % chunk_width];
    if(inner_x < innerspace_width)
    {
        for(uint64_t i = 0; i < innerspace_height; i += chunk_height)
        {
            uint64_t inner_y = i + chunk_y;
            if(inner_y >= innerspace_height)
            {
                break;
            }
            uint64_t idx = inner_y * innerspace_width + inner_x + innerspace_base_index;
            output[idx]  = (algorithm == 2)
                               ? CVT_ACCUM2FLOAT(CVT_FLOAT2ACCUM(input[idx]) - pmax - log(psum))
                               : CVT_ACCUM2FLOAT(exp(CVT_FLOAT2ACCUM(input[idx]) - pmax) / psum);
        }
    }
}

extern "C" __global__ void SoftmaxAccurateFwdDimIsNotLastContiguous(const IO_TYPE* input,
                                                                    IO_TYPE* output,
                                                                    uint64_t reduce_size,
                                                                    uint64_t inner_size,
                                                                    int32_t algorithm)
{
    SoftmaxAccurateFwdDimIsNotLastContiguousKernel<IO_TYPE>(
        input, output, reduce_size, inner_size, algorithm);
}

template <typename TIO>
__device__ void SoftmaxAccurateFwdStrideOneContiguousKernel(const TIO* input,
                                                            TIO* output,
                                                            uint64_t reduce_size,
                                                            int32_t algorithm)
{
    size_t gid       = blockIdx.x;
    size_t tid       = threadIdx.x;
    FLOAT_ACCUM psum = 0;
    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    input  = input + gid * reduce_size;
    output = output + gid * reduce_size;

    // MAX REDUCTION
    FLOAT_ACCUM pmax = std::numeric_limits<FLOAT_ACCUM>::min();
    for(uint64_t i = tid; i < reduce_size; i += LOCAL_SIZE)
    {
        pmax = max(pmax, CVT_FLOAT2ACCUM(input[i]));
    }

    ltmp[tid] = pmax;
    __syncthreads();
    for(uint64_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            ltmp[tid] = max(ltmp[tid], ltmp[tid + i]);
        }
        __syncthreads();
    }

    pmax = ltmp[0];
    __syncthreads();

    // SUM REDUCTION
    for(uint64_t i = tid; i < reduce_size; i += LOCAL_SIZE)
    {
        psum += exp(CVT_FLOAT2ACCUM(input[i]) - pmax);
    }

    ltmp[tid] = psum;
    __syncthreads();
    for(uint64_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            ltmp[tid] += ltmp[tid + i];
        }
        __syncthreads();
    }

    // Final normalization
    psum = ltmp[0];
    for(uint64_t i = tid; i < reduce_size; i += LOCAL_SIZE)
    {
        output[i] = (algorithm == 2)
                        ? CVT_ACCUM2FLOAT(CVT_FLOAT2ACCUM(input[i]) - pmax - log(psum))
                        : CVT_ACCUM2FLOAT(exp(CVT_FLOAT2ACCUM(input[i]) - pmax) / psum);
    }
}

extern "C" __global__ void SoftmaxAccurateFwdStrideOneContiguous(const IO_TYPE* input,
                                                                 IO_TYPE* output,
                                                                 uint64_t reduce_size,
                                                                 int32_t algorithm)
{
    SoftmaxAccurateFwdStrideOneContiguousKernel<IO_TYPE>(input, output, reduce_size, algorithm);
}

template <typename TIO>
__device__ void SoftmaxBwdStrideOneContiguousKernel(const TIO* output,
                                                    const TIO* output_grad,
                                                    TIO* input_grad,
                                                    uint64_t reduce_size,
                                                    int32_t algorithm)
{
    size_t gid       = blockIdx.x;
    size_t tid       = threadIdx.x;
    FLOAT_ACCUM psum = 0;
    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    output      = output + gid * reduce_size;
    output_grad = output_grad + gid * reduce_size;
    input_grad  = input_grad + gid * reduce_size;

    // DOT PRODUCT
    for(uint64_t i = tid; i < reduce_size; i += LOCAL_SIZE)
    {
        if(algorithm == 2)
        {
            psum += CVT_FLOAT2ACCUM(output_grad[i]);
        }
        else
        {
            psum += CVT_FLOAT2ACCUM(output[i]) * CVT_FLOAT2ACCUM(output_grad[i]);
        }
    }

    ltmp[tid] = psum;
    __syncthreads();
    for(uint64_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            ltmp[tid] += ltmp[tid + i];
        }
        __syncthreads();
    }

    // This is the dot product along softmax dim
    psum = ltmp[0];
    // No need for barrier here since this is the last access to local memory

    for(uint64_t i = tid; i < reduce_size; i += LOCAL_SIZE)
    {
        if(algorithm == 2)
        {
            input_grad[i] = CVT_ACCUM2FLOAT(CVT_FLOAT2ACCUM(output_grad[i]) -
                                            psum * exp(CVT_FLOAT2ACCUM(output[i])));
        }
        else
        {
            input_grad[i] = CVT_ACCUM2FLOAT((CVT_FLOAT2ACCUM(output_grad[i]) - psum) *
                                            CVT_FLOAT2ACCUM(output[i]));
        }
    }
}

extern "C" __global__ void SoftmaxBwdStrideOneContiguous(const IO_TYPE* output,
                                                         const IO_TYPE* output_grad,
                                                         IO_TYPE* input_grad,
                                                         uint64_t reduce_size,
                                                         int32_t algorithm)
{
    SoftmaxBwdStrideOneContiguousKernel<IO_TYPE>(
        output, output_grad, input_grad, reduce_size, algorithm);
}

template <typename TIO>
__device__ void SoftmaxBwdSmallContiguousKernel(const TIO* output,
                                                const TIO* output_grad,
                                                TIO* input_grad,
                                                uint64_t reduce_size,
                                                uint64_t inner_size,
                                                uint64_t outer_size,
                                                int32_t algorithm)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= outer_size * inner_size)
        return;
    uint64_t idx_base = (gid / inner_size) * reduce_size * inner_size + gid % inner_size;

    // DOT PRODUCT
    FLOAT_ACCUM psum = 0;
    for(uint64_t i = 0; i < reduce_size; i++)
    {
        uint64_t outgrad_idx = i * inner_size + idx_base;
        uint64_t out_idx     = i * inner_size + idx_base;
        if(algorithm == 2)
        {
            psum += CVT_FLOAT2ACCUM(output_grad[outgrad_idx]);
        }
        else
        {
            psum += CVT_FLOAT2ACCUM(output[out_idx]) * CVT_FLOAT2ACCUM(output_grad[outgrad_idx]);
        }
    }

    // No need for barrier here since this is the last access to local memory

    for(uint64_t i = 0; i < reduce_size; i++)
    {
        uint64_t ingrad_idx  = i * inner_size + idx_base;
        uint64_t outgrad_idx = i * inner_size + idx_base;
        uint64_t out_idx     = i * inner_size + idx_base;
        if(algorithm == 2)
        {
            input_grad[ingrad_idx] = CVT_ACCUM2FLOAT(CVT_FLOAT2ACCUM(output_grad[outgrad_idx]) -
                                                     psum * exp(CVT_FLOAT2ACCUM(output[out_idx])));
        }
        else
        {
            input_grad[ingrad_idx] =
                CVT_ACCUM2FLOAT((CVT_FLOAT2ACCUM(output_grad[outgrad_idx]) - psum) *
                                CVT_FLOAT2ACCUM(output[out_idx]));
        }
    }
}

extern "C" __global__ void SoftmaxBwdSmallContiguous(const IO_TYPE* output,
                                                     const IO_TYPE* output_grad,
                                                     IO_TYPE* input_grad,
                                                     uint64_t reduce_size,
                                                     uint64_t inner_size,
                                                     uint64_t outer_size,
                                                     int32_t algorithm)
{
    SoftmaxBwdSmallContiguousKernel<IO_TYPE>(
        output, output_grad, input_grad, reduce_size, inner_size, outer_size, algorithm);
}

template <typename TIO>
__device__ void SoftmaxBwdDimIsNotLastContiguousKernel(const TIO* output,
                                                       const TIO* output_grad,
                                                       TIO* input_grad,
                                                       uint64_t reduce_size,
                                                       uint64_t inner_size,
                                                       int32_t algorithm)
{
    size_t gid       = blockIdx.x;
    size_t tid       = threadIdx.x;
    FLOAT_ACCUM psum = 0;
    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    uint64_t innerspace_width  = inner_size;
    uint64_t innerspace_height = reduce_size;

    uint64_t chunk_width  = CHUNK_SIZE;
    uint64_t chunk_height = LOCAL_SIZE / chunk_width;

    uint64_t num_chunks_in_innerspace_row = (innerspace_width + chunk_width - 1) / chunk_width;
    uint64_t gid_in_innerspace_row        = (gid % num_chunks_in_innerspace_row) * chunk_width;
    uint64_t innerspace_base_index =
        (gid / num_chunks_in_innerspace_row) * innerspace_height * innerspace_width;

    uint64_t chunk_x = tid % chunk_width;
    uint64_t chunk_y = tid / chunk_width;

    uint64_t inner_x = gid_in_innerspace_row + chunk_x;

    // DOT PRODUCT
    if(inner_x < innerspace_width)
    {
        for(uint64_t i = 0; i < innerspace_height; i += chunk_height)
        {
            uint64_t inner_y = i + chunk_y;
            if(inner_y >= innerspace_height)
            {
                break;
            }
            uint64_t idx = inner_y * innerspace_width + inner_x + innerspace_base_index;
            psum += (algorithm == 2)
                        ? CVT_FLOAT2ACCUM(output_grad[idx])
                        : CVT_FLOAT2ACCUM(output[idx]) * CVT_FLOAT2ACCUM(output_grad[idx]);
        }
    }

    ltmp[tid] = psum;
    __syncthreads();
    for(uint64_t i = LOCAL_SIZE >> 1; i >= chunk_width; i >>= 1)
    {
        if(tid < i)
        {
            ltmp[tid] += ltmp[tid + i];
        }
        __syncthreads();
    }

    // Final normalization
    psum = ltmp[tid % chunk_width];

    if(inner_x < innerspace_width)
    {
        for(uint64_t i = 0; i < innerspace_height; i += chunk_height)
        {
            uint64_t inner_y = i + chunk_y;
            if(inner_y >= innerspace_height)
            {
                break;
            }
            uint64_t idx    = inner_y * innerspace_width + inner_x + innerspace_base_index;
            input_grad[idx] = (algorithm == 2)
                                  ? CVT_ACCUM2FLOAT(CVT_FLOAT2ACCUM(output_grad[idx]) -
                                                    psum * exp(CVT_FLOAT2ACCUM(output[idx])))
                                  : CVT_ACCUM2FLOAT((CVT_FLOAT2ACCUM(output_grad[idx]) - psum) *
                                                    CVT_FLOAT2ACCUM(output[idx]));
        }
    }
}

extern "C" __global__ void SoftmaxBwdDimIsNotLastContiguous(const IO_TYPE* output,
                                                            const IO_TYPE* output_grad,
                                                            IO_TYPE* input_grad,
                                                            uint64_t reduce_size,
                                                            uint64_t inner_size,
                                                            int32_t algorithm)
{
    SoftmaxBwdDimIsNotLastContiguousKernel<IO_TYPE>(
        output, output_grad, input_grad, reduce_size, inner_size, algorithm);
}
