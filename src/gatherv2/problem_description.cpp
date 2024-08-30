
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

#include "miopen/tensor.hpp"
#include <miopen/gatherv2/problem_description.hpp>
#include <numeric>
#include <sstream>

namespace miopen {

namespace gatherv2 {

// M is size of shape
template <int M>
tensor_view_t<M> reshape(const TensorDescriptor& tensorDes, const std::vector<int64_t>& shape)
{
    // check contiguous
    auto tensor   = tensorDes.GetLengths();
    int64_t numel = std::accumulate(tensor.begin(), tensor.end(), 1L, std::multiplies<int64_t>());

    int64_t inferred_dim = -1;
    int64_t new_numel    = 1;
    for(size_t i = 0; i < shape.size(); i++)
    {
        auto dim = shape[i];
        if(dim == -1)
        {
            if(inferred_dim != -1)
            {
                throw std::runtime_error("only one dimension can be inferred");
            }
            inferred_dim = i;
        }
        else
        {
            if(dim <= 0)
            {
                throw std::runtime_error("dimension must be positive");
            }
            new_numel *= dim;
        }
    }
    if(numel < new_numel)
    {
        throw std::runtime_error("invalid shape size");
    }

    tensor_view_t<M> new_tv;
    std::copy(shape.begin(), shape.end(), new_tv.size.begin());
    if(inferred_dim != -1)
    {
        if(numel % new_numel != 0)
        {
            throw std::runtime_error("invalid shape size");
        }
        new_tv.size[inferred_dim] = numel / new_numel;
    }
    else
    {
        if(numel != new_numel)
        {
            throw std::runtime_error("invalid shape size");
        }
    }

    new_tv.get_contiguous_stride();

    return new_tv;
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;

    ss << "gatherv2";

    return NetworkConfig{ss.str()};
}

} // namespace gatherv2

} // namespace miopen