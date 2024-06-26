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

#include "miopen/datatype.hpp"
#include "miopen/errors.hpp"
#include "miopen/miopen.h"
#include <cstdint>
#include <miopen/diag/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace diag {

template <int N>
tensor_view_t<N - 1>
getDiagonal(const tensor_view_t<N>& tv, int64_t offset, int64_t dim1, int64_t dim2)
{
    if(dim1 == dim2)
    {
        MIOPEN_THROW(miopenStatusInternalError, "Diagonal dimensions can not be identical");
    }

    int64_t diag_size;
    if(offset >= 0)
    {
        diag_size = std::max<int64_t>(std::min(tv.size[dim1], tv.size[dim2] - offset), 0);
    }
    else
    {
        diag_size = std::max<int64_t>(std::min(tv.size[dim1] + offset, tv.size[dim2]), 0);
    }

    uint64_t new_offset = tv.offset;
    if(diag_size == 0)
    {
        // skip
    }
    else if(offset >= 0)
    {
        new_offset += offset * tv.stride[dim2];
    }
    else
    {
        new_offset -= offset * tv.stride[dim1];
    }

    std::vector<int64_t> new_sizes(tv.size.begin(), tv.size.end());
    std::vector<int64_t> new_strides(tv.stride.begin(), tv.stride.end());

    new_sizes.erase(new_sizes.begin() + std::max(dim1, dim2));
    new_strides.erase(new_strides.begin() + std::max(dim1, dim2));
    new_sizes.erase(new_sizes.begin() + std::min(dim1, dim2));
    new_strides.erase(new_strides.begin() + std::min(dim1, dim2));
    new_sizes.push_back(diag_size);
    new_strides.push_back(tv.stride[dim1] + tv.stride[dim2]);

    tensor_view_t<N - 1> res;
    res.offset = new_offset;
    res.size   = new_sizes;
    res.stride = new_strides;
    return res;
}

NetworkConfig FwdProblemDescription::MakeNetworkConfig() const
{
    auto inputlength = inputDesc.GetLengths();

    auto input_numel = std::accumulate(
        inputlength.begin(), inputlength.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    auto input_dtype  = miopen::GetDataType(inputDesc.GetType());
    auto output_dtype = miopen::GetDataType(outputDesc.GetType());

    std::ostringstream ss;

    ss << "input_dtype" << input_dtype;
    ss << "output_dtype" << output_dtype;
    ss << "diagonal" << diagonal;
    ss << "numDim" << inputlength.size();
    ss << "input_numel" << input_numel;
    ss << IsAllPacked();

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    auto inputlength = inputGradDesc.GetLengths();

    auto input_numel = std::accumulate(
        inputlength.begin(), inputlength.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    auto input_dtype  = miopen::GetDataType(inputGradDesc.GetType());
    auto output_dtype = miopen::GetDataType(outputGradDesc.GetType());

    std::ostringstream ss;

    ss << "input_dtype" << input_dtype;
    ss << "output_dtype" << output_dtype;
    ss << "diagonal" << diagonal;
    ss << "numDim" << inputlength.size();
    ss << "input_numel" << input_numel;
    ss << IsAllPacked();

    return NetworkConfig{ss.str()};
}

} // namespace diag

} // namespace miopen
