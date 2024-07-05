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

#include "miopen/errors.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <cstdint>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>

#include <string>

namespace miopen {

struct NetworkConfig;

namespace diagonal {

namespace diagembed {

struct FwdProblemDescription : ProblemDescriptionBase
{
    // Forward constructor
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& outputDesc_,
                          int64_t offset_,
                          int64_t dim1_,
                          int64_t dim2_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_), offset(offset_), dim1(dim1_), dim2(dim2_)
    {
        if(dim1 == dim2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "DiagEmbed::FwdProblemDescription: dim1 and dim2 cannot be identical.");
        }

        if(inputDesc.GetLengths().size() > 4)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "DiagEmbed::FwdProblemDescription: Number of tensor dimension must be "
                         "less than 5.");
        }

        if(dim1 < 0 || dim2 < 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "DiagEmbed::FwdProblemDescription: dim1 and dim2 must be non-negative.");
        }

        if(dim1 >= outputDesc.GetLengths().size() || dim2 >= outputDesc.GetLengths().size())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "DiagEmbed::FwdProblemDescription: dim1 and dim2 must be less than the "
                         "number of output tensor's dimension.");
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    int64_t GetOffset() const { return offset; }
    int64_t GetDim1() const { return dim1; }
    int64_t GetDim2() const { return dim2; }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            return false;
        }

        return true;
    }

    bool IsAllPacked() const
    {
        if(!(inputDesc.IsPacked() && outputDesc.IsPacked()))
        {
            return false;
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;

    int64_t offset;
    int64_t dim1;
    int64_t dim2;
};

} // namespace diagembed

} // namespace diagonal

} // namespace miopen
