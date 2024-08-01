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
#include <miopen/var/invoke_params.hpp>
#include <miopen/var/solvers.hpp>
#include <miopen/var.hpp>
#include <miopen/target_properties.hpp>
#include <hip/hip_runtime.h>
#include "../../kernels/tensor_utils.hpp"

#define LOCAL_SIZE 1024

namespace miopen {

namespace solver {

namespace var {

bool VarBackward::IsApplicable([[maybe_unused]] const ExecutionContext& context,
                               const miopen::var::ProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsApplicableSize())
        return false;
    return true;
}

ConvSolution VarBackward::GetSolution([[maybe_unused]] const ExecutionContext& context,
                                      const miopen::var::ProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype            = problem.GetInputDesc().GetType();
    auto input_grad_dims  = problem.GetInputGradDesc().GetLengths();
    auto input_grad_numel = std::accumulate(
        input_grad_dims.begin(), input_grad_dims.end(), 1ULL, std::multiplies<size_t>{});
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

        const auto build_params = KernelBuildParameters{
            {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
            {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
            {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
            {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        };

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::var::InvokeParams>();

            auto input_dims      = params.inputDesc->GetLengths();
            auto input_grad_dims = params.inputGradDesc->GetLengths();
            auto mean_dims       = params.meanDesc->GetLengths();
            auto mean_grad_dims  = params.meanGradDesc->GetLengths();
            auto var_grad_dims   = params.varGradDesc->GetLengths();

            auto input_strides      = params.inputDesc->GetStrides();
            auto input_grad_strides = params.inputGradDesc->GetStrides();
            auto mean_strides       = params.meanDesc->GetStrides();
            auto mean_grad_strides  = params.meanGradDesc->GetStrides();
            auto var_grad_strides   = params.varGradDesc->GetStrides();

            auto dims = *(params.dims);

            auto N = std::accumulate(
                input_grad_dims.begin(), input_grad_dims.end(), 1, std::multiplies<int>{});

            dim_5d_t dims_onehot;
            for(auto dim : dims)
            {
                dims_onehot.x[dim] = 1;
            }

            tensor_view input_tv;
            tensor_view input_grad_tv;
            tensor_view mean_tv;
            tensor_view mean_grad_tv;
            tensor_view var_grad_tv;

            for(int i = 0; i < input_dims.size(); i++)
            {
                input_tv.dimensions[i] = input_dims[i];
                input_tv.strides[i]    = input_strides[i];
            }

            for(int i = 0; i < input_grad_dims.size(); i++)
            {
                input_grad_tv.dimensions[i] = input_grad_dims[i];
                input_grad_tv.strides[i]    = input_grad_strides[i];
            }

            for(int i = 0; i < mean_dims.size(); i++)
            {
                mean_tv.dimensions[i] = mean_dims[i];
                mean_tv.strides[i]    = mean_strides[i];
            }

            for(int i = 0; i < mean_grad_dims.size(); i++)
            {
                mean_grad_tv.dimensions[i] = mean_grad_dims[i];
                mean_grad_tv.strides[i]    = mean_grad_strides[i];
            }

            for(int i = 0; i < var_grad_dims.size(); i++)
            {
                var_grad_tv.dimensions[i] = var_grad_dims[i];
                var_grad_tv.strides[i]    = var_grad_strides[i];
            }

            // Check if input and input_grad tensors are contiguous
            bool is_contiguous = true;
            for(int i = input_dims.size() - 2; i >= 0; --i)
            {
                if(input_strides[i] != input_strides[i + 1] * input_dims[i + 1] ||
                   input_grad_strides[i] != input_grad_strides[i + 1] * input_grad_dims[i + 1])
                {
                    is_contiguous = false;
                    break;
                }
            }

            if(is_contiguous)
            {
                kernel(params.input,
                       params.input_grad,
                       params.mean,
                       params.mean_grad,
                       params.var_grad,
                       N,
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
                       N,
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
