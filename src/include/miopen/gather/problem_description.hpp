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

std::vector<size_t> GetIndicesFlattenShape(const TensorDescriptor& indicesDesc);

struct BwdProblemDescription : ProblemDescriptionBase
{
    // Backward constructor
    BwdProblemDescription(const GatherDescriptor& gatherDesc_,
                          const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& indicesDesc_,
                          const TensorDescriptor& paramGradDesc_)
        : gatherDesc(gatherDesc_),
          outputGradDesc(outputGradDesc_),
          indicesDesc(indicesDesc_),
          paramGradDesc(paramGradDesc_)
    {
        if(indicesDesc.GetNumDims() < 1)
        {
            MIOPEN_THROW("Indices dimension must be >= 1");
        }

        if(outputGradDesc.GetNumDims() < 1)
        {
            MIOPEN_THROW("Output grad dimension must be >= 1");
        }

        if(paramGradDesc.GetNumDims() < 1 || paramGradDesc.GetNumDims() >= 10)
        {
            MIOPEN_THROW("Param grad dimension must be >= 1 and < 10");
        }

        if(indicesDesc.GetType() != miopenInt32 && indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW("Indices must be int32 or int64 tensor");
        }

        size_t outer_dim = indicesDesc.GetNumDims() - 1;

        for(size_t i = 0; i < outer_dim; i++)
        {
            if(indicesDesc.GetLengths()[i] != outputGradDesc.GetLengths()[i])
            {
                MIOPEN_THROW("Outer dimensions of indices and output grad must be equal");
            }
        }

        size_t ix = indicesDesc.GetLengths()[outer_dim];
        if(outputGradDesc.GetNumDims() - outer_dim != paramGradDesc.GetNumDims() - ix)
        {
            MIOPEN_THROW("Inner dimensions of output grad and param grad must be equal");
        }

        for(size_t i = 0; i + outer_dim < outputGradDesc.GetNumDims(); i++)
        {
            if(outputGradDesc.GetLengths()[i + outer_dim] != paramGradDesc.GetLengths()[ix + i])
            {
                MIOPEN_THROW("Dimension " + std::to_string(i + outer_dim) +
                             " of output grad must match dimension " + std::to_string(ix + i) +
                             " of param grad");
            }
        }

        size_t slice_dim = (indicesDesc.GetNumDims() > 1) ? indicesDesc.GetLengths()[outer_dim] : 1;
        size_t batch_dim = (indicesDesc.GetNumDims() > 1) ? indicesDesc.GetNumDims() - 1 : 1;

        if(outputGradDesc.GetNumDims() < batch_dim)
        {
            MIOPEN_THROW("Output grad must have at least " + std::to_string(batch_dim) +
                         " dimensions");
        }

        if(paramGradDesc.GetNumDims() < slice_dim + outputGradDesc.GetNumDims() - batch_dim)
        {
            MIOPEN_THROW("Param grad must have at least " +
                         std::to_string(slice_dim + outputGradDesc.GetNumDims() - batch_dim) +
                         " dimensions");
        }

        if(outputGradDesc.GetNumDims() != batch_dim + paramGradDesc.GetNumDims() - slice_dim)
        {
            MIOPEN_THROW("Invalid output grad and param grad dimensions");
        }

        for(size_t d = 0; d < batch_dim; d++)
        {
            if(outputGradDesc.GetLengths()[d] != indicesDesc.GetLengths()[d])
            {
                MIOPEN_THROW("Dimension " + std::to_string(d) +
                             " of output grad and indices must be equal");
            }
        }

        for(size_t d = 0; d < outputGradDesc.GetNumDims() - batch_dim; d++)
        {
            if(outputGradDesc.GetLengths()[batch_dim + d] !=
               paramGradDesc.GetLengths()[slice_dim + d])
            {
                MIOPEN_THROW("Dimension " + std::to_string(d) +
                             " of output grad and param grad must be equal");
            }
        }

        if(!IsAllContiguous())
        {
            MIOPEN_THROW("All tensors must be contiguous");
        }

        if(!IsSameType())
        {
            MIOPEN_THROW("Output grad and param grad tensors must have the same type");
        }
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }
    const TensorDescriptor& GetParamGradDesc() const { return paramGradDesc; }
    const GatherDescriptor& GetGatherDesc() const { return gatherDesc; }

    bool IsSameType() const;
    bool IsAllContiguous() const;

    NetworkConfig MakeNetworkConfig() const override;

private:
    GatherDescriptor gatherDesc;
    TensorDescriptor outputGradDesc;
    TensorDescriptor indicesDesc;
    TensorDescriptor paramGradDesc;
};

} // namespace gather

} // namespace miopen
