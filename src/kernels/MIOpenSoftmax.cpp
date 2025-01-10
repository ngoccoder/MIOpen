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

#include <limits>
#include "float_types.h"

#define LOCAL_SIZE 256
#define CHUNK_SIZE 64

template <typename TIO>
__device__ void SoftmaxAccurateFwdDimIsNotLastContiguousKernel(const TIO* input,
                                                               TIO* output,
                                                               uint64_t reduce_size,
                                                               uint64_t inner_size,
                                                               uint64_t outer_size,
                                                               uint64_t dim,
                                                               int32_t mode)
{
    size_t gid = blockIdx.x;
    size_t tid = threadIdx.x;

    __shared__ TIO ltmp[LOCAL_SIZE];

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
    TIO pmax = std::numeric_limits<TIO>::min();

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

            pmax = max(pmax, input[idx]);
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
    TIO psum = 0;

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

            psum += exp(input[idx] - pmax);
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
            output[idx] =
                (mode == 1) ? input[idx] - pmax - log(psum) : exp(input[idx] - pmax) / psum;
        }
    }
}

extern "C" __global__ void SoftmaxAccurateFwdDimIsNotLastContiguous(const IO_TYPE* input,
                                                                    IO_TYPE* output,
                                                                    uint64_t reduce_size,
                                                                    uint64_t inner_size,
                                                                    uint64_t outer_size,
                                                                    uint64_t dim,
                                                                    int32_t mode)
{
    SoftmaxAccurateFwdDimIsNotLastContiguousKernel<IO_TYPE>(
        input, output, reduce_size, inner_size, outer_size, dim, mode);
}

template <typename TIO>
__device__ void SoftmaxAccurateFwdStrideOneContiguousKernel(const TIO* input,
                                                            TIO* output,
                                                            uint64_t reduce_size,
                                                            uint64_t inner_size,
                                                            uint64_t outer_size,
                                                            uint64_t dim,
                                                            int32_t mode)
{
    size_t gid = blockIdx.x;
    size_t tid = threadIdx.x;
    TIO psum   = 0;
    __shared__ TIO ltmp[LOCAL_SIZE];

    input  = input + gid * reduce_size;
    output = output + gid * reduce_size;

    // MAX REDUCTION
    TIO pmax = std::numeric_limits<TIO>::min();
    for(uint64_t i = tid; i < reduce_size; i += LOCAL_SIZE)
    {
        pmax = max(pmax, input[i]);
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
        psum += exp(input[i] - pmax);
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
        output[i] = (mode == 1) ? input[i] - pmax - log(psum) : exp(input[i] - pmax) / psum;
    }
}

extern "C" __global__ void SoftmaxAccurateFwdStrideOneContiguous(const IO_TYPE* input,
                                                                 IO_TYPE* output,
                                                                 uint64_t reduce_size,
                                                                 uint64_t inner_size,
                                                                 uint64_t outer_size,
                                                                 uint64_t dim,
                                                                 int32_t mode)
{
    SoftmaxAccurateFwdStrideOneContiguousKernel<IO_TYPE>(
        input, output, reduce_size, inner_size, outer_size, dim, mode);
}

template <typename TIO>
__device__ void SoftmaxBwdStrideOneContiguousKernel(
    const TIO* output, const TIO* output_grad, TIO* input_grad, uint64_t reduce_size, int32_t mode)
{
    size_t gid = blockIdx.x;
    size_t tid = threadIdx.x;
    TIO psum   = 0;
    __shared__ TIO ltmp[LOCAL_SIZE];

    output      = output + gid * reduce_size;
    output_grad = output_grad + gid * reduce_size;
    input_grad  = input_grad + gid * reduce_size;

    // DOT PRODUCT
    for(uint64_t i = tid; i < reduce_size; i += LOCAL_SIZE)
    {
        if(mode == 1)
        {
            psum += output_grad[i];
        }
        else
        {
            psum += output[i] * output_grad[i];
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
        if(mode == 1)
        {
            input_grad[i] = output_grad[i] - psum * exp(output[i]);
        }
        else
        {
            input_grad[i] = output_grad[i] - psum * output[i];
        }
    }
}

extern "C" __global__ void SoftmaxBwdStrideOneContiguous(const IO_TYPE* output,
                                                         const IO_TYPE* output_grad,
                                                         IO_TYPE* input_grad,
                                                         uint64_t reduce_size,
                                                         int32_t mode)
{
    SoftmaxBwdStrideOneContiguousKernel<IO_TYPE>(
        output, output_grad, input_grad, reduce_size, mode);
}

template <typename TIO>
__device__ void SoftmaxBwdSmallContiguousKernel(const TIO* output,
                                                const TIO* output_grad,
                                                TIO* input_grad,
                                                uint64_t reduce_size,
                                                uint64_t inner_size,
                                                uint64_t outer_size,
                                                int32_t mode)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= outer_size * inner_size)
        return;
    uint64_t idx_base = (gid / inner_size) * reduce_size * inner_size + gid % inner_size;

    // DOT PRODUCT
    TIO psum = 0;
    for(uint64_t i = 0; i < reduce_size; i++)
    {
        uint64_t outgrad_idx = i * inner_size + idx_base;
        uint64_t out_idx     = i * inner_size + idx_base;
        if(mode == 1)
        {
            psum += output_grad[outgrad_idx];
        }
        else
        {
            psum += output[out_idx] * output_grad[outgrad_idx];
        }
    }

    // No need for barrier here since this is the last access to local memory

    for(uint64_t i = 0; i < reduce_size; i++)
    {
        uint64_t input_idx   = i * inner_size + idx_base;
        uint64_t outgrad_idx = i * inner_size + idx_base;
        uint64_t out_idx     = i * inner_size + idx_base;
        if(mode == 1)
        {
            input_grad[outgrad_idx] = output_grad[outgrad_idx] - psum * exp(output[out_idx]);
        }
        else
        {
            input_grad[outgrad_idx] = output_grad[outgrad_idx] - psum * output[out_idx];
        }
    }
}

extern "C" __global__ void SoftmaxBwdSmallContiguous(const IO_TYPE* output,
                                                     const IO_TYPE* output_grad,
                                                     IO_TYPE* input_grad,
                                                     uint64_t reduce_size,
                                                     uint64_t inner_size,
                                                     uint64_t outer_size,
                                                     int32_t mode)
{
    SoftmaxBwdSmallContiguousKernel<IO_TYPE>(
        output, output_grad, input_grad, reduce_size, inner_size, outer_size, mode);
}

template <typename TIO>
__device__ void SoftmaxBwdDimIsNotLastContiguousKernel(const TIO* output,
                                                       const TIO* output_grad,
                                                       TIO* input_grad,
                                                       uint64_t reduce_size,
                                                       uint64_t inner_size,
                                                       uint64_t outer_size,
                                                       int32_t mode)
{
    size_t gid = blockIdx.x;
    size_t tid = threadIdx.x;
    TIO psum   = 0;
    __shared__ TIO ltmp[LOCAL_SIZE];

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
            psum += (mode == 1) ? output_grad[idx] : output[idx] * output_grad[idx];
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
            input_grad[idx] = (mode == 1) ? output_grad[idx] - psum * exp(output[idx])
                                          : (output_grad[idx] - psum) * output[idx];
        }
    }
}

extern "C" __global__ void SoftmaxBwdDimIsNotLastContiguous(const IO_TYPE* output,
                                                            const IO_TYPE* output_grad,
                                                            IO_TYPE* input_grad,
                                                            uint64_t reduce_size,
                                                            uint64_t inner_size,
                                                            uint64_t outer_size,
                                                            int32_t mode)
{
    SoftmaxBwdDimIsNotLastContiguousKernel<IO_TYPE>(
        output, output_grad, input_grad, reduce_size, inner_size, outer_size, mode);
}
