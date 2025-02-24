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

#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/var.hpp>

extern "C" miopenStatus_t miopenVarBackward(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t inputDesc,
                                            const void* input,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            void* input_grad,
                                            const miopenTensorDescriptor_t meanDesc,
                                            const void* mean,
                                            const miopenTensorDescriptor_t meanGradDesc,
                                            const void* mean_grad,
                                            const miopenTensorDescriptor_t varGradDesc,
                                            const void* var_grad,
                                            const int* dims,
                                            int num_dims,
                                            bool keepdim,
                                            bool unbiased,
                                            int divisor)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        inputGradDesc,
                        input_grad,
                        meanDesc,
                        mean,
                        meanGradDesc,
                        mean_grad,
                        varGradDesc,
                        var_grad,
                        dims,
                        num_dims,
                        keepdim,
                        unbiased,
                        divisor);

    return miopen::try_([&] {
        miopen::var::VarBackward(miopen::deref(handle),
                                 miopen::deref(inputDesc),
                                 DataCast(input),
                                 miopen::deref(inputGradDesc),
                                 DataCast(input_grad),
                                 miopen::deref(meanDesc),
                                 DataCast(mean),
                                 miopen::deref(meanGradDesc),
                                 DataCast(mean_grad),
                                 miopen::deref(varGradDesc),
                                 DataCast(var_grad),
                                 dims,
                                 num_dims,
                                 keepdim,
                                 unbiased,
                                 divisor);
    });
}
