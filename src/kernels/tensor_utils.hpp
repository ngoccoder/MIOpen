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

struct tensor_view
{
    uint64_t dimensions[5] = {1, 1, 1, 1, 1};
    uint64_t strides[5]    = {1, 1, 1, 1, 1};
};

struct dim_5d_t
{
    uint64_t x[5] = {0, 0, 0, 0, 0};
};

__device__ void inline GET_NCDHW(uint64_t ncdhw[5], uint64_t id, const uint64_t dimensions[5])
{
    uint64_t ncdh = (id) / dimensions[4];
    ncdhw[4]      = (id) % dimensions[4];
    uint64_t ncd  = (ncdh) / dimensions[3];
    ncdhw[3]      = (ncdh) % dimensions[3];
    uint64_t nc   = (ncd) / dimensions[2];
    ncdhw[2]      = (ncd) % dimensions[2];
    uint64_t n    = (nc) / dimensions[1];
    ncdhw[1]      = (nc) % dimensions[1];
    ncdhw[0]      = n;
}

__device__ uint64_t inline GET_STRIDED_INDEX(const uint64_t indices[5], const uint64_t strides[5])
{
    return indices[0] * strides[0] + indices[1] * strides[1] + indices[2] * strides[2] +
           indices[3] * strides[3] + indices[4] * strides[4];
}
