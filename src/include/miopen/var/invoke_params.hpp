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

#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace var {

struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams(const TensorDescriptor& inputDesc_,
                 ConstData_t input_,
                 const TensorDescriptor& inputGradDesc_,
                 Data_t input_grad_,
                 const TensorDescriptor& meanDesc_,
                 ConstData_t mean_,
                 const TensorDescriptor& meanGradDesc_,
                 ConstData_t mean_grad_,
                 const TensorDescriptor& varGradDesc_,
                 ConstData_t var_grad_,
                 const std::vector<int>& dims_,
                 const bool keepdim_,
                 const bool unbiased_,
                 const int divisor_)
        : inputDesc(inputDesc_),
          input(input_),
          inputGradDesc(inputGradDesc_),
          input_grad(input_grad_),
          meanDesc(meanDesc_),
          mean(mean_),
          meanGradDesc(meanGradDesc_),
          mean_grad(mean_grad_),
          varGradDesc(varGradDesc_),
          var_grad(var_grad_),
          dims(dims_),
          keepdim(keepdim_),
          unbiased(unbiased_),
          divisor(divisor_)
    {
    }

    const TensorDescriptor& inputDesc     = nullptr;
    const TensorDescriptor& inputGradDesc = nullptr;
    const TensorDescriptor& meanDesc      = nullptr;
    const TensorDescriptor& meanGradDesc  = nullptr;
    const TensorDescriptor& varGradDesc   = nullptr;

    ConstData_t input     = nullptr;
    Data_t input_grad     = nullptr;
    ConstData_t mean      = nullptr;
    ConstData_t mean_grad = nullptr;
    ConstData_t var_grad  = nullptr;

    const std::vector<int>& dims = nullptr;
    const bool keepdim;
    const bool unbiased;
    const int divisor;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
}

} // namespace var

} // namespace miopen