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
#include <miopen/cosinesimilarity.hpp>
#include <miopen/cosinesimilarity/invoke_params.hpp>
#include <miopen/cosinesimilarity/problem_description.hpp>
#include <miopen/cosinesimilarity/solvers.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace cosinesimilarity {

miopenStatus_t CosineSimilarityForward(Handle& handle,
                                       const TensorDescriptor& input1Desc,
                                       ConstData_t input1,
                                       const TensorDescriptor& input2Desc,
                                       ConstData_t input2,
                                       const TensorDescriptor& outputDesc,
                                       Data_t output,
                                       uint32_t dim,
                                       float eps)
{
    const auto problem = miopen::cosinesimilarity::FwdProblemDescription{
        input1Desc, input2Desc, outputDesc, dim, eps};

    const auto invoke_params = [&]() {
        auto tmp       = cosinesimilarity::FwdInvokeParams{};
        tmp.type       = InvokeType::Run;
        tmp.input1Desc = &input1Desc;
        tmp.input2Desc = &input2Desc;
        tmp.outputDesc = &outputDesc;
        tmp.input1     = input1;
        tmp.input2     = input2;
        tmp.output     = output;
        tmp.dim        = dim;
        tmp.eps        = eps;
        return tmp;
    }();

    const auto algo = AlgorithmName{"CosineSimilarityForward"};
    const auto solvers =
        solver::SolverContainer<solver::cosinesimilarity::CosineSimilarityForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t CosineSimilarityBackward(Handle& handle,
                                        const TensorDescriptor& input1Desc,
                                        ConstData_t input1,
                                        const TensorDescriptor& input2Desc,
                                        ConstData_t input2,
                                        const TensorDescriptor& outputGradDesc,
                                        ConstData_t outputGrad,
                                        const TensorDescriptor& input1GradDesc,
                                        Data_t input1Grad,
                                        const TensorDescriptor& input2GradDesc,
                                        Data_t input2Grad,
                                        uint32_t dim,
                                        float eps)
{
    const auto problem = miopen::cosinesimilarity::BwdProblemDescription {}
}

} // namespace cosinesimilarity

} // namespace miopen
