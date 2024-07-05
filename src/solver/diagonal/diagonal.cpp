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

#include "miopen/tensor_view_utils.hpp"

namespace miopen {

namespace solver {

namespace diagonal {

tensor_view_t<5>
getDiagonal(const TensorDescriptor& tensor, int64_t offset, int64_t dim1, int64_t dim2)
{
    if(dim1 == dim2)
    {
        MIOPEN_THROW(miopenStatusInternalError, "Diagonal dimensions can not be identical");
    }

    int64_t diag_size;
    auto lens     = tensor.GetLengths();
    size_t dimNum = lens.size();
    auto strides  = tensor.GetStrides();
    if(offset >= 0)
    {
        diag_size = std::max<int64_t>(std::min(lens[dim1], lens[dim2] - offset), 0);
    }
    else
    {
        diag_size = std::max<int64_t>(std::min(lens[dim1] + offset, lens[dim2]), 0);
    }

    uint64_t new_offset = 0;
    if(diag_size == 0)
    {
        // skip
    }
    else if(offset >= 0)
    {
        new_offset += offset * strides[dim2];
    }
    else
    {
        new_offset -= offset * strides[dim1];
    }

    tensor_view_t<5> res;
    res.offset = new_offset;

    int curIdx    = 0;
    int curNewIdx = 0;
    while(curNewIdx < dimNum - 2)
    {
        if(curIdx == dim1 || curIdx == dim2)
        {
            curIdx++;
        }
        else
        {
            res.size[curNewIdx]   = lens[curIdx];
            res.stride[curNewIdx] = strides[curIdx];
            curNewIdx++;
            curIdx++;
        }
    }
    res.size[dimNum - 2]   = diag_size;
    res.stride[dimNum - 2] = strides[dim1] + strides[dim2];

    for(int i = dimNum - 1; i < 5; ++i)
    {
        res.stride[i] = (i == 0 ? 1 : res.stride[i - 1]);
        res.size[i]   = 1;
    }

    return res;
}

} // namespace diagonal

} // namespace solver

} // namespace miopen
