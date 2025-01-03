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

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace embeddingbag {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& weightDesc_,
                          const TensorDescriptor& offsetsDesc_,
                          const TensorDescriptor& perSampleWeightDesc_,
                          const TensorDescriptor& outputDesc_,
                          const miopenEmbeddingBagMode_t mode_)
        : inputDesc(inputDesc_),
          weightDesc(weightDesc_),
          offsetsDesc(offsetsDesc_),
          perSampleWeightDesc(perSampleWeightDesc_),
          outputDesc(outputDesc_),
          mode(mode_)
    {
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetOffsetsDesc() const { return offsetsDesc; }
    const TensorDescriptor& GetPerSampleWeightDesc() const { return perSampleWeightDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    miopenEmbeddingBagMode_t GetMode() const { return mode; }

    bool IsSameType() const { return outputDesc.GetType() == weightDesc.GetType(); }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor offsetsDesc;
    TensorDescriptor perSampleWeightDesc;
    TensorDescriptor outputDesc;
    miopenEmbeddingBagMode_t mode;
};

} // namespace embeddingbag

} // namespace miopen
