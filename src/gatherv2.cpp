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

#include <miopen/gatherv2/invoke_params.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/gatherv2/solvers.hpp>
#include <miopen/gatherv2.hpp>

namespace miopen {

miopenStatus_t GatherV2Backward(Handle& handle,
                                const TensorDescriptor& outputGradDesc,
                                ConstData_t outputgrad,
                                const TensorDescriptor& indiceDesc,
                                ConstData_t indices,
                                const TensorDescriptor& paramGradDesc,
                                Data_t paramGrad,
                                int64_t axis,
                                int batch_dims)
{
    const auto problem = gatherv2::BwdProblemDescription{
        outputGradDesc, indiceDesc, paramGradDesc, axis, batch_dims};

    const auto invoke_params = [&]() {
        auto tmp           = gatherv2::BwdInvokeParams{};
        tmp.outputGradDesc = &outputGradDesc;
        tmp.indicesDesc    = &indiceDesc;
        tmp.paramGradDesc  = &paramGradDesc;
        tmp.outputGrad     = outputgrad;
        tmp.indices        = indices;
        tmp.paramGrad      = paramGrad;
        tmp.axis           = axis;
        tmp.batch_dims     = batch_dims;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"GatherV2Backward"};
    const auto solvers = solver::SolverContainer<solver::gatherv2::GatherV2Backward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
