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
#include <miopen/var/invoke_params.hpp>
#include <miopen/var/solvers.hpp>
#include <miopen/var.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t VarBackward(Handle& handle,
                           const TensorDescriptor& inputDesc,
                           ConstData_t input,
                           const TensorDescriptor& inputGradDesc,
                           Data_t input_grad,
                           const TensorDescriptor& meanDesc,
                           ConstData_t mean,
                           const TensorDescriptor& meanGradDesc,
                           ConstData_t mean_grad,
                           const TensorDescriptor& varGradDesc,
                           ConstData_t var_grad,
                           const int* dims,
                           int num_dims,
                           bool keepdim,
                           bool unbiased,
                           int divisor)
{
    std::vector dims_vector(dims, dims + num_dims);

    const auto problem       = var::ProblemDescription{inputDesc,
                                                 inputGradDesc,
                                                 meanDesc,
                                                 meanGradDesc,
                                                 varGradDesc,
                                                 dims_vector,
                                                 keepdim,
                                                 unbiased,
                                                 divisor};
    const auto invoke_params = var::InvokeParams{inputDesc,
                                                 input,
                                                 inputGradDesc,
                                                 input_grad,
                                                 meanDesc,
                                                 mean,
                                                 meanGradDesc,
                                                 mean_grad,
                                                 varGradDesc,
                                                 var_grad,
                                                 dims_vector,
                                                 keepdim,
                                                 unbiased,
                                                 divisor};
    const auto algo          = AlgorithmName{"VarBackward"};
    const auto solvers       = solver::SolverContainer<solver::var::VarBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen
