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

#include "miopen/trace/problem_description.hpp"

#include <miopen/common.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/trace.hpp>
#include <miopen/tensor.hpp>
#include <miopen/trace/invoke_params.hpp>
#include <miopen/trace/solvers.hpp>

namespace miopen {

namespace trace {

size_t GetTraceForwardWorkspaceSize(Handle& handle,
                                    const TensorDescriptor& inputDesc,
                                    const TensorDescriptor& outputDesc)
{
    auto ctx           = ExecutionContext{&handle};
    const auto problem = trace::FwdProblemDescription{inputDesc, outputDesc};

    const auto algo    = AlgorithmName{"TraceForward"};
    const auto solvers = solver::SolverContainer<solver::trace::TraceForward>{};

    auto pair_size_vector = solvers.GetWorkspaceSizes(ctx, problem);

    return pair_size_vector.empty() ? static_cast<size_t>(-1) : pair_size_vector.front().second;
}

miopenStatus_t TraceForward(Handle& handle,
                            Data_t workspace,
                            size_t workspaceSizeInBytes,
                            const TensorDescriptor& inputDesc,
                            ConstData_t input,
                            const TensorDescriptor& outputDesc,
                            Data_t output)
{
    const auto problem = trace::FwdProblemDescription{inputDesc, outputDesc};

    const auto invoke_params = [&]() {
        auto tmp           = trace::FwdInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.inputDesc      = &inputDesc;
        tmp.outputDesc     = &outputDesc;
        tmp.input          = input;
        tmp.output         = output;
        tmp.workspace      = workspace;
        tmp.workspace_size = workspaceSizeInBytes;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"TraceForward"};
    const auto solvers = solver::SolverContainer<solver::trace::TraceForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t TraceBackward(Handle& handle,
                             const TensorDescriptor& outputGradDesc,
                             ConstData_t outputGrad,
                             const TensorDescriptor& inputGradDesc,
                             Data_t inputGrad)
{
    const auto problem = miopen::trace::BwdProblemDescription{outputGradDesc, inputGradDesc};

    const auto invoke_params = [&]() {
        auto tmp           = trace::BwdInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.inputGrad      = inputGrad;
        tmp.outputGrad     = outputGrad;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"TraceBackward"};
    const auto solvers = solver::SolverContainer<solver::trace::TraceBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace trace

} // namespace miopen
