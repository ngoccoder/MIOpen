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
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/problem_description_base.hpp>
#include "../src/kernels/tensor_view.hpp"

namespace miopen {

struct NetworkConfig;

namespace gather {

template <int M>
tensor_view_t<M> reshape(const TensorDescriptor& tensorDes, const std::vector<size_t>& shape);

extern template tensor_view_t<3> reshape<3>(const TensorDescriptor& tensorDes,
                                            const std::vector<size_t>& shape);

extern template tensor_view_t<4> reshape<4>(const TensorDescriptor& tensorDes,
                                            const std::vector<size_t>& shape);

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
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const GatherDescriptor& GetGatherDesc() const { return gatherDesc; }

    bool IsSameType() const;
    bool IsAllContiguous() const;

    NetworkConfig MakeNetworkConfig() const override;

private:
    GatherDescriptor gatherDesc;
    TensorDescriptor inputDesc;
    TensorDescriptor indicesDesc;
    TensorDescriptor outputDesc;
};

} // namespace gather

} // namespace miopen
