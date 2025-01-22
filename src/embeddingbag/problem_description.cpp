/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/embeddingbag/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace embeddingbag {

NetworkConfig FwdProblemDescription::MakeNetworkConfig() const
{
    auto weight_dtype  = weightDesc.GetType();
    auto output_numel  = outputDesc.GetElementSize();
    auto weight_len    = weightDesc.GetLengths();
    auto offsets_numel = offsetsDesc.GetElementSize();

    std::ostringstream ss;

    ss << "embeddingbag_fwd";
    ss << "weight_dtype" << weight_dtype;
    ss << "num_embedding" << weight_len[0];
    ss << "embedding_dim" << weight_len[1];
    ss << "output_size" << output_numel;
    ss << "mode" << mode;
    ss << "offsets_size" << offsets_numel;

    return NetworkConfig{ss.str()};
}

} // namespace embeddingbag

} // namespace miopen
