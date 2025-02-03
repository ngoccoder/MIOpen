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

#include <cstdint>
#include <miopen/common.hpp>
#include <miopen/cosinesimilarity.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenCosineSimilarityForward(miopenHandle_t handle,
                                                        const miopenTensorDescriptor_t input1Desc,
                                                        const void* input1,
                                                        const miopenTensorDescriptor_t input2Desc,
                                                        const void* input2,
                                                        const miopenTensorDescriptor_t outputDesc,
                                                        void* output,
                                                        uint32_t dim,
                                                        float eps)
{
    MIOPEN_LOG_FUNCTION(
        handle, input1Desc, input1, input2Desc, input2, outputDesc, output, dim, eps);
    return miopen::try_([&] {
        miopen::cosinesimilarity::CosineSimilarityForward(miopen::deref(handle),
                                                          miopen::deref(input1Desc),
                                                          DataCast(input1),
                                                          miopen::deref(input2Desc),
                                                          DataCast(input2),
                                                          miopen::deref(outputDesc),
                                                          DataCast(output),
                                                          dim,
                                                          eps);
    });
}

extern "C" miopenStatus_t
miopenCosineSimilarityBackward(miopenHandle_t handle,
                               const miopenTensorDescriptor_t input1Desc,
                               const void* input1,
                               const miopenTensorDescriptor_t input2Desc,
                               const void* input2,
                               const miopenTensorDescriptor_t outputGradDesc,
                               const void* outputGrad,
                               const miopenTensorDescriptor_t input1GradDesc,
                               void* input1Grad,
                               const miopenTensorDescriptor_t input2GradDesc,
                               void* input2Grad,
                               uint32_t dim,
                               float eps)
{
    MIOPEN_LOG_FUNCTION(handle,
                        input1Desc,
                        input1,
                        input2Desc,
                        input2,
                        outputGradDesc,
                        outputGrad,
                        input1GradDesc,
                        input1Grad,
                        input2GradDesc,
                        input2Grad,
                        dim,
                        eps);
    return miopen::try_([&] {
        miopen::cosinesimilarity::CosineSimilarityBackward(miopen::deref(handle),
                                                           miopen::deref(input1Desc),
                                                           DataCast(input1),
                                                           miopen::deref(input2Desc),
                                                           DataCast(input2),
                                                           miopen::deref(outputGradDesc),
                                                           DataCast(outputGrad),
                                                           miopen::deref(input1GradDesc),
                                                           DataCast(input1Grad),
                                                           miopen::deref(input2GradDesc),
                                                           DataCast(input2Grad),
                                                           dim,
                                                           eps);
    });
}
