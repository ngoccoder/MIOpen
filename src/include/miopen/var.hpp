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

#include <miopen/common.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

namespace var {

MIOPEN_INTERNALS_EXPORT miopenStatus_t VarBackward(Handle& handle,
                                                   const TensorDescriptor& inputDesc,
                                                   ConstData_t input,
                                                   const TensorDescriptor& inputGradDesc,
                                                   Data_t input_grad,
                                                   const TensorDescriptor& meanDesc,
                                                   ConstData_t mean,
                                                   const TensorDescriptor& meanGradDesc,
                                                   ConstData_t mean_grad,
                                                   const TensorDescriptor& varGradDesc,
                                                   ConstData_t var_grad,
                                                   const int* dims,
                                                   uint32_t num_dims,
                                                   bool keepdim,
                                                   bool unbiased,
                                                   uint32_t divisor);

} // namespace var

} // namespace miopen
