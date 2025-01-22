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

#include <miopen/common.hpp>
#include <miopen/embeddingbag.hpp>
#include <miopen/embeddingbag/invoke_params.hpp>
#include <miopen/embeddingbag/problem_description.hpp>
#include <miopen/embeddingbag/solvers.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace embeddingbag {

miopenStatus_t EmbeddingBagForward(Handle& handle,
                                   const TensorDescriptor& inputDesc,
                                   ConstData_t input,
                                   const TensorDescriptor& weightDesc,
                                   ConstData_t weight,
                                   const TensorDescriptor& offsetsDesc,
                                   ConstData_t offsets,
                                   const TensorDescriptor& perSampleWeightDesc,
                                   ConstData_t perSampleWeight,
                                   const TensorDescriptor& outputDesc,
                                   Data_t output,
                                   miopenEmbeddingBagMode_t mode)
{
    const auto problem = embeddingbag::FwdProblemDescription{
        inputDesc, weightDesc, offsetsDesc, perSampleWeightDesc, outputDesc, mode};

    const auto invoke_params = [&]() {
        auto tmp                = embeddingbag::FwdInvokeParams{};
        tmp.type                = InvokeType::Run;
        tmp.inputDesc           = &inputDesc;
        tmp.weightDesc          = &weightDesc;
        tmp.offsetsDesc         = &offsetsDesc;
        tmp.perSampleWeightDesc = &perSampleWeightDesc;
        tmp.outputDesc          = &outputDesc;
        tmp.input               = input;
        tmp.weight              = weight;
        tmp.offsets             = offsets;
        tmp.perSampleWeight     = perSampleWeight;
        tmp.output              = output;
        tmp.mode                = mode;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"EmbeddingBagForward"};
    const auto solvers = solver::SolverContainer<solver::embeddingbag::EmbeddingBagForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace embeddingbag

} // namespace miopen
