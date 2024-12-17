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

#include "miopen/errors.hpp"
#include <miopen/miopen.h>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace trace {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_, const TensorDescriptor& outputDesc_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_)
    {
        if(inputDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Input tensor must be 2D.");
        }

        if(outputDesc.GetNumDims() != 1)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Output tensor must be 1D.");
        }

        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Input and output tensor must have same type.");
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& inputGradDesc_)
        : outputGradDesc(outputGradDesc_), inputGradDesc(inputGradDesc_)
    {
        if(inputGradDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Input grad tensor must be 2D.");
        }

        if(outputGradDesc.GetNumDims() != 1)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Output grad tensor must be 1D.");
        }

        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Input grad and output grad tensor must have same type.");
        }
    }

    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor outputGradDesc;
    TensorDescriptor inputGradDesc;
};

} // namespace trace

} // namespace miopen
