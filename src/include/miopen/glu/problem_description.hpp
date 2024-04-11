/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

namespace glu {

enum class Direction
{
    Forward,
    Backward,
};

struct ProblemDescription : ProblemDescriptionBase
{
    // Forward constructor
    ProblemDescription(const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       int32_t dim_)
        : direction(Direction::Forward), xDesc(xDesc_), yDesc(yDesc_), dim(dim_)
    {
        if(xDesc.GetLengths().size() != yDesc.GetLengths().size())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GLU::ProblemDescription: Number of tensor dimension do not match.");
        }
        if(xDesc.GetLengths()[dim] % 2 != 0)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GLU::ProblemDescription: The split dimension size of input tensor should "
                         "be divisible by 2.");
        }
    }

    Direction GetDirection() const { return direction; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    int32_t GetDim() const { return dim; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightLength() const
    {
        for(int32_t i = 0; i < xDesc.GetLengths().size(); i++)
        {
            if (i == dim) {
                if (xDesc.GetLengths()[i] / 2 != yDesc.GetLengths()[i]) {
                    return false;
                }
            } else {
                if (xDesc.GetLengths()[i] != yDesc.GetLengths()[i]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool IsRightDim() const
    {
        if((dim < 0) || (dim > xDesc.GetLengths().size()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(
                miopenStatusBadParm,
                "Reduce: is greater than 0 and less than or equal tensor dimension length.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && yDesc.IsPacked()))
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Unpacked tensors not supported.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsNotLastDim() const
    {
        if(dim == xDesc.GetLengths().size() - 1)
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    Direction direction;
    TensorDescriptor xDesc;
    TensorDescriptor yDesc;
    
    int32_t dim;
};

} // namespace glu

} // namespace miopen
