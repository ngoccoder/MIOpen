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

namespace var {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& inputGradDesc_,
                       const TensorDescriptor& meanDesc_,
                       const TensorDescriptor& meanGradDesc_,
                       const TensorDescriptor& varGradDesc_,
                       const std::vector<int>& dims_,
                       const bool keepdim_,
                       const bool unbiased_,
                       const int divisor_)
        : inputDesc(inputDesc_),
          inputGradDesc(inputGradDesc_),
          meanDesc(meanDesc_),
          meanGradDesc(meanGradDesc_),
          varGradDesc(varGradDesc_),
          dims(dims_),
          keepdim(keepdim_),
          unbiased(unbiased_),
          divisor(divisor_)
    {
        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "All tensors must have the same data type.");
        }

        if(!IsApplicableSize())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Input gradient size must be in range [1024, 1024 * 1024 * 2].");
        }

        if(!IsAllPacked())
        {
            MIOPEN_THROW(miopenStatusBadParm, "All tensors must be packed.");
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetMeanDesc() const { return meanDesc; }
    const TensorDescriptor& GetMeanGradDesc() const { return meanGradDesc; }
    const TensorDescriptor& GetVarGradDesc() const { return varGradDesc; }

    const std::vector<int>& GetDims() const { return dims; }
    bool GetKeepDim() const { return keepdim; }
    bool GetUnbiased() const { return unbiased; }
    uint32_t GetDivisor() const { return divisor; }

    bool IsSameType() const
    {
        return inputDesc.GetType() == inputGradDesc.GetType() &&
               inputDesc.GetType() == meanDesc.GetType() &&
               inputDesc.GetType() == meanGradDesc.GetType() &&
               inputDesc.GetType() == varGradDesc.GetType();
    }

    bool IsAllPacked() const
    {
        if(inputDesc.IsPacked() && inputGradDesc.IsPacked() && meanDesc.IsPacked() &&
           meanGradDesc.IsPacked() && varGradDesc.IsPacked())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool IsApplicableSize() const
    {
        auto input_grad_numel = inputGradDesc.GetElementSize();

        if(input_grad_numel >= 1024 && input_grad_numel <= 1024 * 1024 * 2)
            return true;
        return false;
    }

    bool IsAllContiguous() const
    {
        return inputDesc.IsContiguous() && inputGradDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor inputGradDesc;
    TensorDescriptor meanDesc;
    TensorDescriptor meanGradDesc;
    TensorDescriptor varGradDesc;

    std::vector<int> dims;
    bool keepdim;
    bool unbiased;
    uint32_t divisor;
};

} // namespace var

} // namespace miopen
