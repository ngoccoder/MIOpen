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

#include <cstdint>
#include <miopen/common.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace embedding {

struct BwdInvokeParams : public miopen::InvokeParams
{
    BwdInvokeParams() = default;

    const TensorDescriptor* inputDesc      = nullptr;
    const TensorDescriptor* outputGradDesc = nullptr;
    const TensorDescriptor* weightGradDesc = nullptr;

    ConstData_t input       = nullptr;
    ConstData_t outputGrad  = nullptr;
    Data_t weightGrad       = nullptr;
    ConstData_t scale_freq  = nullptr;
    bool scale_grad_by_freq = false;
    int64_t padding_idx     = 0;

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }
};

} // namespace embedding

} // namespace miopen
