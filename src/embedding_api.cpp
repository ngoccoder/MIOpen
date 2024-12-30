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

#include <cstdint>
#include <miopen/common.hpp>
#include <miopen/embedding.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenEmbeddingBackward(miopenHandle_t handle,
                                                  const miopenTensorDescriptor_t inputDesc,
                                                  const void* input,
                                                  const miopenTensorDescriptor_t outputGradDesc,
                                                  const void* outputGrad,
                                                  const miopenTensorDescriptor_t weightGradDesc,
                                                  void* weightGrad,
                                                  const void* indices_freq,
                                                  int64_t padding_idx)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        outputGradDesc,
                        outputGrad,
                        weightGradDesc,
                        weightGrad,
                        indices_freq,
                        padding_idx);
    return miopen::try_([&] {
        miopen::embedding::EmbeddingBackward(miopen::deref(handle),
                                             miopen::deref(inputDesc),
                                             DataCast(input),
                                             miopen::deref(outputGradDesc),
                                             DataCast(outputGrad),
                                             miopen::deref(weightGradDesc),
                                             DataCast(weightGrad),
                                             DataCast(indices_freq),
                                             padding_idx);
    });
}
