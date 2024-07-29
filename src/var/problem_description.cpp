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

#include <miopen/var/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace var {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;
    ss << "varbwd-";

    const auto& input_length      = inputDesc.GetLengths();
    const auto& input_grad_length = inputGradDesc.GetLengths();
    const auto& mean_length       = meanDesc.GetLengths();
    const auto& mean_grad_length  = meanGradDesc.GetLengths();
    const auto& var_grad_length   = varGradDesc.GetLengths();
    const auto& is_all_contiguous = IsAllContiguous();

    ss << "input-";
    for(auto len : input_length)
    {
        ss << len << "-";
    }

    ss << "input_grad-";
    for(auto len : input_grad_length)
    {
        ss << len << "-";
    }

    ss << "mean-";
    for(auto len : mean_length)
    {
        ss << len << "-";
    }

    ss << "mean_grad-";
    for(auto len : mean_grad_length)
    {
        ss << len << "-";
    }

    ss << "var_grad-";
    for(auto len : var_grad_length)
    {
        ss << len << "-";
    }

    ss << "dims-";
    for(auto dim : dims)
    {
        ss << dim << "-";
    }

    ss << "keepdim-" << keepdim << "-";
    ss << "unbiased-" << unbiased << "-";
    ss << "divisor-" << divisor << "-";
    ss << "all_contiguous-" << is_all_contiguous;

    return NetworkConfig{ss.str()};
}

} // namespace var

} // namespace miopen
