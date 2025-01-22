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

#include <miopen/embeddingbag.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_ops.hpp>

#define CHECK_DESC_EXIST(desc) (((desc) != nullptr) ? miopen::deref(desc) : dummyDesc)

extern "C" miopenStatus_t
miopenEmbeddingBagForward(miopenHandle_t handle,
                          const miopenTensorDescriptor_t inputDesc,
                          const void* input,
                          const miopenTensorDescriptor_t weightDesc,
                          const void* weight,
                          const miopenTensorDescriptor_t offsetsDesc,
                          const void* offsets,
                          const miopenTensorDescriptor_t perSampleWeightDesc,
                          const void* perSampleWeight,
                          const miopenTensorDescriptor_t outputDesc,
                          void* output,
                          miopenEmbeddingBagMode_t mode)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        weightDesc,
                        weight,
                        offsetsDesc,
                        offsets,
                        perSampleWeightDesc,
                        perSampleWeight,
                        outputDesc,
                        output,
                        mode);
    const miopen::TensorDescriptor dummyDesc;

    return miopen::try_([&] {
        miopen::embeddingbag::EmbeddingBagForward(miopen::deref(handle),
                                                  miopen::deref(inputDesc),
                                                  DataCast(input),
                                                  miopen::deref(weightDesc),
                                                  DataCast(weight),
                                                  CHECK_DESC_EXIST(offsetsDesc),
                                                  DataCast(offsets),
                                                  CHECK_DESC_EXIST(perSampleWeightDesc),
                                                  DataCast(perSampleWeight),
                                                  miopen::deref(outputDesc),
                                                  DataCast(output),
                                                  mode);
    });
}
