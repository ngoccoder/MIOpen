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

#include <miopen/gather.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

extern "C" miopenStatus_t miopenCreateGatherDescriptor(miopenGatherDescriptor_t* gatherDesc)
{
    MIOPEN_LOG_FUNCTION(gatherDesc);
    return miopen::try_([&] {
        auto& desc = miopen::deref(gatherDesc);
        desc       = new miopen::GatherDescriptor();
    });
}

extern "C" miopenStatus_t miopenSetGatherDescriptor(const miopenGatherDescriptor_t gatherDesc,
                                                    miopenGatherMode_t mode,
                                                    uint32_t dim,
                                                    uint32_t batch_dims)
{
    MIOPEN_LOG_FUNCTION(gatherDesc, mode, dim, batch_dims);
    return miopen::try_([&] {
        auto& desc = miopen::deref(gatherDesc);
        desc       = miopen::GatherDescriptor(mode, dim, batch_dims);
    });
}

extern "C" miopenStatus_t miopenGetGatherDescriptor(const miopenGatherDescriptor_t gatherDesc,
                                                    miopenGatherMode_t* mode,
                                                    uint32_t* dim,
                                                    uint32_t* batch_dims)
{
    MIOPEN_LOG_FUNCTION(gatherDesc);
    return miopen::try_([&] {
        *mode       = miopen::deref(gatherDesc).getMode();
        *dim        = miopen::deref(gatherDesc).getDim();
        *batch_dims = miopen::deref(gatherDesc).getBatchDims();
    });
}

extern "C" miopenStatus_t miopenGatherForward(miopenHandle_t handle,
                                              const miopenGatherDescriptor_t gatherDesc,
                                              const miopenTensorDescriptor_t inputDesc,
                                              const void* input,
                                              const miopenTensorDescriptor_t indicesDesc,
                                              const void* indices,
                                              const miopenTensorDescriptor_t outputDesc,
                                              void* output)
{
    MIOPEN_LOG_FUNCTION(
        handle, gatherDesc, inputDesc, input, indicesDesc, indices, outputDesc, output);

    return miopen::try_([&] {
        miopen::deref(gatherDesc)
            .Forward(miopen::deref(handle),
                     miopen::deref(inputDesc),
                     DataCast(input),
                     miopen::deref(indicesDesc),
                     DataCast(indices),
                     miopen::deref(outputDesc),
                     DataCast(output));
    });
}

extern "C" miopenStatus_t miopenDestroyGatherDescriptor(miopenGatherDescriptor_t gatherDesc)
{
    MIOPEN_LOG_FUNCTION(gatherDesc);
    return miopen::try_([&] { miopen_destroy_object(gatherDesc); });
}
