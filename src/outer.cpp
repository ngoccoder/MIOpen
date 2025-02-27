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

#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/outer/invoke_params.hpp>
#include <miopen/outer/solvers.hpp>
#include <miopen/outer.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

namespace outer {

miopenStatus_t OuterForward(Handle& handle,
                            const TensorDescriptor& x1Desc,
                            ConstData_t x1,
                            const TensorDescriptor& x2Desc,
                            ConstData_t x2,
                            const TensorDescriptor& yDesc,
                            Data_t y)
{
    const auto problem = outer::FwdProblemDescription(x1Desc, x2Desc, yDesc);

    const auto invoke_params = [&]() {
        auto result   = outer::FwdInvokeParams{};
        result.x1Desc = &x1Desc;
        result.x2Desc = &x2Desc;
        result.yDesc  = &yDesc;
        result.x1     = x1;
        result.x2     = x2;
        result.y      = y;
        return result;
    }();

    const auto algo    = AlgorithmName{"OuterForward"};
    const auto solvers = solver::SolverContainer<solver::outer::OuterForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t OuterBackward(Handle& handle,
                             const TensorDescriptor& x1Desc,
                             ConstData_t x1,
                             const TensorDescriptor& x2Desc,
                             ConstData_t x2,
                             const TensorDescriptor& x1GradDesc,
                             Data_t x1Grad,
                             const TensorDescriptor& x2GradDesc,
                             Data_t x2Grad,
                             const TensorDescriptor& yGradDesc,
                             ConstData_t yGrad)
{
    const auto problem =
        outer::BwdProblemDescription(x1Desc, x2Desc, x1GradDesc, x2GradDesc, yGradDesc);
    const auto invoke_params = [&]() {
        auto result       = outer::BwdInvokeParams{};
        result.x1Desc     = &x1Desc;
        result.x2Desc     = &x2Desc;
        result.x1GradDesc = &x1GradDesc;
        result.x2GradDesc = &x2GradDesc;
        result.yGradDesc  = &yGradDesc;
        result.x1         = x1;
        result.x2         = x2;
        result.x1Grad     = x1Grad;
        result.x2Grad     = x2Grad;
        result.yGrad      = yGrad;
        return result;
    }();

    const auto algo    = AlgorithmName{"OuterBackward"};
    const auto solvers = solver::SolverContainer<solver::outer::OuterBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace outer

} // namespace miopen
