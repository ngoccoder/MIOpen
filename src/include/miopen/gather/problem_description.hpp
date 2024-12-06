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
#include <cstdint>

#include <miopen/errors.hpp>
#include <miopen/gather.hpp>
#include <miopen/tensor.hpp>
#include <miopen/problem_description_base.hpp>
#include "../src/kernels/tensor_view.hpp"
#include "miopen/miopen.h"

namespace miopen {

struct NetworkConfig;

namespace gather {

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
