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

#include <cstddef>

#include <miopen/errors.hpp>
#include <miopen/gather.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/problem_description_base.hpp>

namespace miopen {

struct NetworkConfig;

namespace gather {

struct FwdProblemDescription : ProblemDescriptionBase
{
    // Forward constructor
    FwdProblemDescription(const GatherDescriptor& gatherDesc_,
                          const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& indicesDesc_,
                          const TensorDescriptor& outputDesc_)
        : gatherDesc(gatherDesc_),
          inputDesc(inputDesc_),
          indicesDesc(indicesDesc_),
          outputDesc(outputDesc_)
    {
        if(gatherDesc.getDim() >= inputDesc.GetNumDims())
        {
            MIOPEN_THROW("Gather: Dimension out of range");
        }

        if(inputDesc.GetNumDims() != indicesDesc.GetNumDims())
        {
            MIOPEN_THROW("Input and indices dimension size should be the same");
        }

        if(indicesDesc.GetNumDims() != outputDesc.GetNumDims())
        {
            MIOPEN_THROW("Indices and output dimension size should be the same");
        }

        for(size_t i = 0; i < indicesDesc.GetNumDims(); i++)
        {
            if(i == gatherDesc.getDim())
                continue;
            if(indicesDesc.GetLengths()[i] > inputDesc.GetLengths()[i])
            {
                MIOPEN_THROW("Index size of dimension " + std::to_string(i) + " out of bound.");
            }
        }

        if(indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW("Index tensor type should be int64");
        }

        if(inputDesc.GetNumDims() > 5 || indicesDesc.GetNumDims() > 5 ||
           outputDesc.GetNumDims() > 5)
        {
            MIOPEN_THROW("Gather only supports up to 5 dimensions");
        }

        if(!IsSameType())
        {
            MIOPEN_THROW("Input and output tensor type should be the same");
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const GatherDescriptor& GetGatherDesc() const { return gatherDesc; }

    bool IsSameType() const;

    NetworkConfig MakeNetworkConfig() const override;

private:
    GatherDescriptor gatherDesc;
    TensorDescriptor inputDesc;
    TensorDescriptor indicesDesc;
    TensorDescriptor outputDesc;
};

} // namespace gather

} // namespace miopen
