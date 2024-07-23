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

#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <cassert>
#include <string>

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
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetMeanDesc() const { return meanDesc; }
    const TensorDescriptor& GetMeanGradDesc() const { return meanGradDesc; }
    const TensorDescriptor& GetVarGradDesc() const { return varGradDesc; }

    const std::vector<int>& GetDims() const { return dims; }
    bool GetKeepDim() const { return keepdim; }
    bool GetUnbiased() const { return unbiased; }
    int GetDivisor() const { return divisor; }

    bool IsSameType() const
    {
        if(inputDesc.GetType() == inputGradDesc.GetType() &&
           inputDesc.GetType() == meanDesc.GetType() &&
           inputDesc.GetType() == meanGradDesc.GetType() &&
           inputDesc.GetType() == varGradDesc.GetType())
        {
            return true;
        }
        else
        {
            return false;
        }
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
        int input_grad_numel = std::accumulate(inputGradDesc.GetLengths().begin(),
                                               inputGradDesc.GetLengths().end(),
                                               1,
                                               std::multiplies<int>());

        if(input_grad_numel >= 1024 && input_grad_numel <= 1024 * 128)
            return true;
        return false;
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
    int divisor;
};

} // namespace var

} // namespace miopen
