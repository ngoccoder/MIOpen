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

#include <cstdint>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>

#include <string>

namespace miopen {

struct NetworkConfig;

namespace diag {

enum class Direction
{
    Forward,
    Backward,
};

struct ProblemDescription : ProblemDescriptionBase
{
    // Forward constructor
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& outputDesc_,
                       int32_t diagonal_)
        : direction(Direction::Forward),
          inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          diagonal(diagonal_)
    {
        if(inputDesc.GetLengths().size() != 1 && inputDesc.GetLengths().size() != 2)
        {

            MIOPEN_THROW(miopenStatusBadParm,
                         "Diag::ProblemDescription: Number of tensor dimension is not 1 or 2.");
        }
    }

    Direction GetDirection() const { return direction; }
    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    int32_t GetDiagonal() const { return diagonal; }

    bool IsSameType() const
    {
        if(direction == Direction::Forward)
        {
            if(inputDesc.GetType() != outputDesc.GetType())
            {
                return false;
            }
        }

        return true;
    }

    bool IsAllPacked() const
    {
        if(direction == Direction::Forward)
        {
            if(!(inputDesc.IsPacked() && outputDesc.IsPacked()))
            {
                return false;
            }
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    Direction direction;
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;

    int32_t diagonal;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace diag

} // namespace miopen
