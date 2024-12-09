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

#include <miopen/errors.hpp>
#include <miopen/gather/problem_description.hpp>
#include <miopen/tensor.hpp>

#include <sstream>
#include <vector>

namespace miopen {

namespace gather {

// indices.shape = [num_indices, idx_depth]
std::vector<size_t> GetIndicesFlattenShape(const TensorDescriptor& indicesDesc)
{
    int indices_dim = 2;
    std::vector<size_t> indices_flatten_shape(indices_dim);
    int offset = indicesDesc.GetNumDims() - indices_dim;
    for(int out_dim = indices_dim - 1; out_dim >= 0; out_dim--)
    {
        const int in_dim               = out_dim + offset;
        indices_flatten_shape[out_dim] = in_dim < 0 ? 1 : indicesDesc.GetLengths()[in_dim];
    }
    for(int in_dim = 0; in_dim < offset; in_dim++)
    {
        indices_flatten_shape[0] *= indicesDesc.GetLengths()[in_dim];
    }

    return indices_flatten_shape;
}

bool BwdProblemDescription::IsSameType() const
{
    if(outputGradDesc.GetType() != paramGradDesc.GetType())
    {
        return false;
    }

    return true;
}

bool BwdProblemDescription::IsAllContiguous() const
{
    if(outputGradDesc.IsContiguous() && indicesDesc.IsContiguous() && paramGradDesc.IsContiguous())
    {
        return true;
    }
    return false;
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;

    ss << "gathernd";
    ss << "dtype" << paramGradDesc.GetType();
    ss << "index_type" << indicesDesc.GetType();
    ss << "param_size" << paramGradDesc.GetElementSize();
    ss << "outgrad_size" << outputGradDesc.GetElementSize();
    ss << "mode " << gatherDesc.getMode();
    ss << "dim " << gatherDesc.getDim();
    ss << "batch dim " << gatherDesc.getBatchDims();

    return NetworkConfig{ss.str()};
}

} // namespace gather

} // namespace miopen
