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
#pragma once

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace cosinesimilarity {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& input1Desc_,
                          const TensorDescriptor& input2Desc_,
                          const TensorDescriptor& outputDesc_,
                          uint32_t dim_,
                          float eps_)
        : input1Desc(input1Desc_),
          input2Desc(input2Desc_),
          outputDesc(outputDesc_),
          dim(dim_),
          eps(eps_)
    {
        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Input and output tensor types do not match");
        }
    }

    const TensorDescriptor& GetInput1Desc() const { return input1Desc; }
    const TensorDescriptor& GetInput2Desc() const { return input2Desc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    bool IsSameType() const
    {
        return input1Desc.GetType() == input2Desc.GetType() &&
               input1Desc.GetType() == outputDesc.GetType();
    }

    // bool isAllContiguous() const
    //{
    //    return inputDesc.IsContiguous() && outputGradDesc.IsContiguous() &&
    //           weightGradDesc.IsContiguous();
    //}

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor input1Desc;
    TensorDescriptor input2Desc;
    TensorDescriptor outputDesc;

    uint32_t dim;
    float eps;
};

} // namespace cosinesimilarity

} // namespace miopen
