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
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/var/invoke_params.hpp>
#include <miopen/var/solvers.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace var {

bool VarBackward::IsApplicable(const ExecutionContext& /*context*/,
                               const miopen::var::ProblemDescription& problem) const
{
    if(!(problem.GetInputDesc().GetType() == miopenFloat ||
         problem.GetInputDesc().GetType() == miopenHalf ||
         problem.GetInputDesc().GetType() == miopenBFloat16))
        return false;

    return true;
}

ConvSolution VarBackward::GetSolution(const ExecutionContext& /*context*/,
                                      const miopen::var::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype             = problem.GetInputDesc().GetType();
    auto io_dtype          = miopen::GetDataType(dtype);
    auto input_grad_dims   = problem.GetInputGradDesc().GetLengths();
    auto input_grad_numel  = problem.GetInputGradDesc().GetElementSize();
    auto is_all_contiguous = problem.IsAllContiguous();

    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(input_grad_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        auto kernel = KernelInfo{};

        if(is_all_contiguous)
        {
            kernel.kernel_file = "MIOpenVar.cpp";
            kernel.kernel_name = "VarBackwardContiguous";
        }
        else
        {
            kernel.kernel_file = "MIOpenVar.cpp";
            kernel.kernel_name = "VarBackward";
        }

        const auto build_params =
            KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                                  {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                                  {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                                  {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype}};

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [input_grad_numel,
                              is_all_contiguous](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::var::InvokeParams>();

            auto dims = params.dims;
            dim_5d_t dims_onehot;
            for(auto dim : dims)
            {
                dims_onehot.x[dim] = 1;
            }

            tensor_view_t<5> input_tv = get_inner_expanded_tv<5>(miopen::deref(params.inputDesc));
            tensor_view_t<5> input_grad_tv =
                get_inner_expanded_tv<5>(miopen::deref(params.inputGradDesc));
            tensor_view_t<5> mean_tv = get_inner_expanded_tv<5>(miopen::deref(params.meanDesc));
            tensor_view_t<5> mean_grad_tv =
                get_inner_expanded_tv<5>(miopen::deref(params.meanGradDesc));
            tensor_view_t<5> var_grad_tv =
                get_inner_expanded_tv<5>(miopen::deref(params.varGradDesc));

            if(is_all_contiguous)
            {
                kernel(params.input,
                       params.input_grad,
                       params.mean,
                       params.mean_grad,
                       params.var_grad,
                       input_grad_numel,
                       dims_onehot,
                       params.unbiased,
                       params.divisor,
                       input_grad_tv,
                       mean_tv,
                       mean_grad_tv,
                       var_grad_tv);
            }
            else
            {
                kernel(params.input,
                       params.input_grad,
                       params.mean,
                       params.mean_grad,
                       params.var_grad,
                       input_grad_numel,
                       dims_onehot,
                       params.unbiased,
                       params.divisor,
                       input_tv,
                       input_grad_tv,
                       mean_tv,
                       mean_grad_tv,
                       var_grad_tv);
            }
        };
    };

    return result;
}

} // namespace var

} // namespace solver

} // namespace miopen
