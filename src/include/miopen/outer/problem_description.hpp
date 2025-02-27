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

#include <miopen/miopen.h>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace outer {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& x1Desc_,
                          const TensorDescriptor& x2Desc_,
                          const TensorDescriptor& yDesc_)
        : x1Desc(x1Desc_), x2Desc(x2Desc_), yDesc(yDesc_)
    {
        if(x1Desc.GetNumDims() != 1 || x2Desc.GetNumDims() != 1 || yDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Outer: Tensor dimensions are not valid.");
        }

        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Outer: Tensor types do not match.");
        }
    }

    const TensorDescriptor& GetX1Desc() const { return x1Desc; }
    const TensorDescriptor& GetX2Desc() const { return x2Desc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    bool IsSameType() const
    {
        return x1Desc.GetType() == x2Desc.GetType() && x1Desc.GetType() == yDesc.GetType();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor x1Desc;
    TensorDescriptor x2Desc;
    TensorDescriptor yDesc;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& x1Desc_,
                          const TensorDescriptor& x2Desc_,
                          const TensorDescriptor& x1GradDesc_,
                          const TensorDescriptor& x2GradDesc_,
                          const TensorDescriptor& yGradDesc_)
        : x1Desc(x1Desc_),
          x2Desc(x2Desc_),
          x1GradDesc(x1GradDesc_),
          x2GradDesc(x2GradDesc_),
          yGradDesc(yGradDesc_)
    {
        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Outer: Tensor types do not match.");
        }

        if(x1Desc.GetNumDims() != 1 || x2Desc.GetNumDims() != 1 || x1GradDesc.GetNumDims() != 1 ||
           x2GradDesc.GetNumDims() != 1 || yGradDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Outer: Tensor dimensions are not valid.");
        }
    }

    const TensorDescriptor& GetX1Desc() const { return x1Desc; }
    const TensorDescriptor& GetX2Desc() const { return x2Desc; }
    const TensorDescriptor& GetX1GradDesc() const { return x1GradDesc; }
    const TensorDescriptor& GetX2GradDesc() const { return x2GradDesc; }
    const TensorDescriptor& GetYGradDesc() const { return yGradDesc; }

    bool IsSameType() const
    {
        return x1Desc.GetType() == x2Desc.GetType() && x1Desc.GetType() == x1GradDesc.GetType() &&
               x1Desc.GetType() == x2GradDesc.GetType() && x1Desc.GetType() == yGradDesc.GetType();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor x1Desc;
    TensorDescriptor x2Desc;
    TensorDescriptor x1GradDesc;
    TensorDescriptor x2GradDesc;
    TensorDescriptor yGradDesc;
};

} // namespace outer

} // namespace miopen
