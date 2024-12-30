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

#include <miopen/common.hpp>
#include <miopen/embedding.hpp>
#include <miopen/embedding/invoke_params.hpp>
#include <miopen/embedding/problem_description.hpp>
#include <miopen/embedding/solvers.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace embedding {

miopenStatus_t EmbeddingBackward(Handle& handle,
                                 const TensorDescriptor& inputDesc,
                                 ConstData_t input,
                                 const TensorDescriptor& outputGradDesc,
                                 ConstData_t outputGrad,
                                 const TensorDescriptor& weightGradDesc,
                                 Data_t weightGrad,
                                 ConstData_t indices_freq,
                                 int64_t padding_idx)
{
    const auto problem =
        miopen::embedding::BwdProblemDescription{inputDesc, outputGradDesc, weightGradDesc};

    const auto invoke_params = [&]() {
        auto tmp           = embedding::BwdInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.inputDesc      = &inputDesc;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.weightGradDesc = &weightGradDesc;
        tmp.input          = input;
        tmp.outputGrad     = outputGrad;
        tmp.weightGrad     = weightGrad;
        tmp.indices_freq   = indices_freq;
        tmp.padding_idx    = padding_idx;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"EmbeddingBackward"};
    const auto solvers = solver::SolverContainer<solver::embedding::EmbeddingBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace embedding

} // namespace miopen
