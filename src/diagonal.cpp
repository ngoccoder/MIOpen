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

#include <miopen/diagonal/diag/invoke_params.hpp>
#include "miopen/common.hpp"
#include "miopen/diagonal/diagembed/invoke_params.hpp"
#include "miopen/diagonal/diagflat/invoke_params.hpp"
#include "miopen/handle.hpp"
#include "miopen/miopen.h"
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/diagonal/solvers.hpp>
#include <miopen/diagonal.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t DiagForward(Handle& handle,
                           const TensorDescriptor& inputDesc,
                           ConstData_t input,
                           const TensorDescriptor& outputDesc,
                           Data_t output,
                           int64_t diagonal)
{
    const auto problem = diagonal::diag::FwdProblemDescription{inputDesc, outputDesc, diagonal};

    const auto invoke_params = [&]() {
        auto tmp       = diagonal::diag::FwdInvokeParams{};
        tmp.type       = InvokeType::Run;
        tmp.inputDesc  = &inputDesc;
        tmp.input      = input;
        tmp.outputDesc = &outputDesc;
        tmp.output     = output;
        tmp.diagonal   = diagonal;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"DiagForward"};
    const auto solvers = solver::SolverContainer<solver::diagonal::diag::DiagForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t DiagFlatForward(Handle& handle,
                               const TensorDescriptor& inputDesc,
                               ConstData_t input,
                               const TensorDescriptor& outputDesc,
                               Data_t output,
                               int64_t offset)
{
    const auto problem = diagonal::diagflat::FwdProblemDescription{inputDesc, outputDesc, offset};

    const auto invoke_params = [&]() {
        auto tmp       = diagonal::diagflat::FwdInvokeParams{};
        tmp.type       = InvokeType::Run;
        tmp.inputDesc  = &inputDesc;
        tmp.input      = input;
        tmp.outputDesc = &outputDesc;
        tmp.output     = output;
        tmp.offset     = offset;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"DiagFlatForward"};
    const auto solvers = solver::SolverContainer<solver::diagonal::diagflat::DiagFlatForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t DiagEmbedForward(Handle& handle,
                                const TensorDescriptor& inputDesc,
                                ConstData_t input,
                                const TensorDescriptor& outputDesc,
                                Data_t output,
                                int64_t offset,
                                int64_t dim1,
                                int64_t dim2)
{
    const auto problem =
        diagonal::diagembed::FwdProblemDescription{inputDesc, outputDesc, offset, dim1, dim2};

    const auto invoke_params = [&]() {
        auto tmp       = diagonal::diagembed::FwdInvokeParams{};
        tmp.type       = InvokeType::Run;
        tmp.inputDesc  = &inputDesc;
        tmp.input      = input;
        tmp.outputDesc = &outputDesc;
        tmp.output     = output;
        tmp.offset     = offset;
        tmp.dim1       = dim1;
        tmp.dim2       = dim2;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"DiagEmbededForward"};
    const auto solvers = solver::SolverContainer<solver::diagonal::diagembed::DiagEmbedForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
