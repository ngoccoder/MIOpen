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

#include <miopen/var.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdVarBackward(const miopenTensorDescriptor_t inputGradDesc,
                              const int* dims,
                              const int num_dims,
                              const bool keepdim,
                              const bool unbiased,
                              const int divisor)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputGradDesc).GetType();
        if(dtype == miopenFloat)
        {
            ss << "varfp32";
        }
        else if(dtype == miopenHalf)
        {
            ss << "varfp16";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "varbf16";
        }

        int32_t size = {0};
        miopenGetTensorDescriptorSize(inputGradDesc, &size);
        ss << " -n " << miopen::dref(inputGradDesc).GetLengths()[0];
        if(size == 5)
        {
            ss << " -c " << miopen::dref(inputGradDesc).GetLengths()[1] << " -D "
               << miopen::dref(inputGradDesc).GetLengths()[2] << " -H "
               << miopen::dref(inputGradDesc).GetLengths()[3] << " -W "
               << miopen::dref(inputGradDesc).GetLengths()[4];
        }
        else if(size == 4)
        {
            ss << " -c " << miopen::dref(inputGradDesc).GetLengths()[1] << " -D "
               << miopen::dref(inputGradDesc).GetLengths()[2] << " -H "
               << miopen::dref(inputGradDesc).GetLengths()[3]
        }
        else if(size == 3)
        {
            ss << " -c " << miopen::dref(inputGradDesc).GetLengths()[1] << " -D "
               << miopen::dref(inputGradDesc).GetLengths()[2];
        }
        else if(size == 2)
        {
            ss << " -c " << miopen::dref(inputGradDesc).GetLengths()[1];
        }

        ss << " -dims ";
        for(int i = 0; i < num_dims; i++)
        {
            ss << dims[i] << " ";
        }

        ss << " -keepdim " << ((keepdim) ? "true" : "false") << " -unbiased "
           << ((unbiased) ? "true" : "false") << " -divisor " << divisor;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
};

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
                                            const int num_dims,
                                            const bool keepdim,
                                            const bool unbiased,
                                            const int divisor)
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

    LogCmdVarBackward(inputGradDesc, dims, num_dims, keepdim, unbiased, divisor);

    return miopen::try_([&] {
        miopen::VarBackward(miopen::deref(handle),
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