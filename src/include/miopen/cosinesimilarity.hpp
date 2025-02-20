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

#include <miopen/common.hpp>
#include <miopen/export_internals.h>

namespace miopen {

struct Handle;
struct TensorDescriptor;

namespace cosinesimilarity {

MIOPEN_INTERNALS_EXPORT miopenStatus_t CosineSimilarityForward(Handle& handle,
                                                               const TensorDescriptor& input1Desc,
                                                               ConstData_t input1,
                                                               const TensorDescriptor& input2Desc,
                                                               ConstData_t input2,
                                                               const TensorDescriptor& outputDesc,
                                                               Data_t output,
                                                               uint32_t dim,
                                                               float eps);

MIOPEN_INTERNALS_EXPORT miopenStatus_t
CosineSimilarityBackward(Handle& handle,
                         const TensorDescriptor& input1Desc,
                         ConstData_t input1,
                         const TensorDescriptor& input2Desc,
                         ConstData_t input2,
                         const TensorDescriptor& outputGradDesc,
                         ConstData_t outputGrad,
                         const TensorDescriptor& input1GradDesc,
                         Data_t input1Grad,
                         const TensorDescriptor& input2GradDesc,
                         Data_t input2Grad,
                         uint32_t dim,
                         float eps);

} // namespace cosinesimilarity

} // namespace miopen
