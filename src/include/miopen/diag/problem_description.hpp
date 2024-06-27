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

#include "miopen/tensor_view_utils.hpp"
#include <cstdint>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>

#include <string>

namespace miopen {

struct NetworkConfig;

namespace diag {

template <int N>
tensor_view_t<N - 1>
getDiagonal(const tensor_view_t<N>& tv, int64_t offset, int64_t dim1, int64_t dim2);
extern template tensor_view_t<1>
getDiagonal(const tensor_view_t<2>& tv, int64_t offset, int64_t dim1, int64_t dim2);

struct FwdProblemDescription : ProblemDescriptionBase
{
    // Forward constructor
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& outputDesc_,
                          int32_t diagonal_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_), diagonal(diagonal_)
    {
        if(inputDesc.GetLengths().size() != 1 && inputDesc.GetLengths().size() != 2)
        {

            MIOPEN_THROW(miopenStatusBadParm,
                         "Diag::FwdProblemDescription: Number of tensor dimension is not 1 or 2.");
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    int64_t GetDiagonal() const { return diagonal; }

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

    int64_t diagonal;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    // Forward constructor
    BwdProblemDescription(const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& inputGradDesc_,
                          int64_t diagonal_)
        : outputGradDesc(outputGradDesc_), inputGradDesc(inputGradDesc_), diagonal(diagonal_)
    {
        if(inputGradDesc.GetLengths().size() != 1 && inputGradDesc.GetLengths().size() != 2)
        {

            MIOPEN_THROW(miopenStatusBadParm,
                         "Diag::BwdProblemDescription: Number of tensor dimension is not 1 or 2.");
        }
    }

    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    int64_t GetDiagonal() const { return diagonal; }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            return false;
        }

        return true;
    }

    bool IsAllPacked() const
    {
        if(!(inputGradDesc.IsPacked() && outputGradDesc.IsPacked()))
        {
            return false;
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor outputGradDesc;
    TensorDescriptor inputGradDesc;

    int64_t diagonal;
};

} // namespace diag

} // namespace miopen
