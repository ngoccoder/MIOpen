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

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace embedding {

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& weightGradDesc_)
        : inputDesc(inputDesc_), outputGradDesc(outputGradDesc_), weightGradDesc(weightGradDesc_)
    {
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetWeightGradDesc() const { return weightGradDesc; }

    bool IsSameType() const
    {
        if(outputGradDesc.GetType() != weightGradDesc.GetType())
        {
            return false;
        }
        return true;
    }

    bool isAllContiguous() const
    {
        return inputDesc.IsContiguous() && outputGradDesc.IsContiguous() &&
               weightGradDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor outputGradDesc;
    TensorDescriptor weightGradDesc;
};

} // namespace embedding

} // namespace miopen
