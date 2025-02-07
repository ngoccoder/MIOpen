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
#include <miopen/tensor.hpp>
#include <miopen/problem_description_base.hpp>
#include "../src/kernels/tensor_view.hpp"

namespace miopen {

struct NetworkConfig;

namespace gatherv2 {

template <int M>
tensor_view_t<M> reshape(const TensorDescriptor& tensorDes, const std::vector<int64_t>& shape);

struct BwdProblemDescription : ProblemDescriptionBase
{
    // Backward constructor
    BwdProblemDescription(const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& indicesDesc_,
                          const TensorDescriptor& paramGradDesc_,
                          int64_t axis_,
                          int batch_dims_)
        : outputGradDesc(outputGradDesc_),
          indicesDesc(indicesDesc_),
          paramGradDesc(paramGradDesc_),
          axis(axis_),
          batch_dims(batch_dims_)
    {
        if(indicesDesc.GetType() != miopenInt32 && indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "GatherV2::BwdProblemDescription: Indices tensor must be int32 or int64 tensor.");
        }

        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GatherV2::BwdProblemDescription: Output grad and param grad tensors must "
                         "be same type.");
        }
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }
    const TensorDescriptor& GetParamGradDesc() const { return paramGradDesc; }
    int64_t GetAxis() const { return axis; }
    int GetBatchDims() const { return batch_dims; }

    bool IsSameType() const
    {
        if(outputGradDesc.GetType() != paramGradDesc.GetType())
        {
            return false;
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor outputGradDesc;
    TensorDescriptor indicesDesc;
    TensorDescriptor paramGradDesc;

    int64_t axis;
    int batch_dims;
};

} // namespace gatherv2

} // namespace miopen
