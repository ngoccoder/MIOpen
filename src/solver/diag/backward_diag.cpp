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

#include "miopen/diag/problem_description.hpp"
#include "miopen/kernel_info.hpp"
#include "miopen/mlo_internal.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <cstddef>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/diag/invoke_params.hpp>
#include <miopen/diag/solvers.hpp>
#include <miopen/diag.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace diag {

bool DiagBackward::IsApplicable(const ExecutionContext& context,
                                const miopen::diag::BwdProblemDescription& problem) const
{
    std::ignore    = context;
    auto inputDims = problem.GetInputGradDesc().GetLengths();

    if(!problem.IsSameType())
        return false;
    if(!problem.IsAllPacked())
        return false;
    return true;
}

ConvSolution DiagBackward::GetSolution(const ExecutionContext& context,
                                       const miopen::diag::BwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype         = problem.GetInputGradDesc().GetType();
    auto input_dtype   = miopen::GetDataType(problem.GetInputGradDesc().GetType());
    auto output_dtype  = miopen::GetDataType(problem.GetOutputGradDesc().GetType());
    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenDiag.cpp";

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
        {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    auto inLens = problem.GetInputGradDesc().GetLengths();
    if(inLens.size() == 1)
    {
        int64_t ingrad_numel = problem.GetInputGradDesc().GetElementSize();

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(ingrad_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        kernel.kernel_name = "Diag2dForward";

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);

        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::diag::BwdInvokeParams>();
                auto ingrad_numel     = params.inputGradDesc->GetElementSize();
                auto outgrad_tv       = get_inner_expanded_tv<2>(*params.outputGradDesc);
                long offset = (params.diagonal >= 0 ? params.diagonal * outgrad_tv.stride[1]
                                                    : -params.diagonal * outgrad_tv.stride[0]);

                kernel(params.outputGrad, params.inputGrad, ingrad_numel, offset, outgrad_tv);
            };
        };
    }
    else if(inLens[0] == inLens[1])
    {
        auto outgrad_numel = problem.GetOutputGradDesc().GetElementSize();

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(outgrad_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        kernel.kernel_name = "Diag1dForward";

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);

        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::diag::BwdInvokeParams>();
                auto outgrad_numel    = params.outputGradDesc->GetElementSize();
                auto inputgrad_tv     = get_inner_expanded_tv<2>(*params.inputGradDesc);
                long offset = (params.diagonal >= 0 ? params.diagonal * inputgrad_tv.stride[1]
                                                    : -params.diagonal * inputgrad_tv.stride[0]);

                kernel(params.outputGrad, params.inputGrad, outgrad_numel, offset, inputgrad_tv);
            };
        };
    }
    else
    {
        auto outgrad_numel = problem.GetOutputGradDesc().GetElementSize();

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(outgrad_numel, xlocalsize);
        size_t ylocalsize = 1;
        size_t ygridsize  = 1;
        size_t zlocalsize = 1;
        size_t zgridsize  = 1;

        kernel.kernel_name = "Assign1d";

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(ylocalsize);
        kernel.l_wk.push_back(zlocalsize);

        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(ygridsize);
        kernel.g_wk.push_back(zgridsize);

        result.construction_params.push_back(kernel);

        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::diag::BwdInvokeParams>();
                auto outgrad_numel    = params.outputGradDesc->GetElementSize();
                auto outgrad_tv       = get_inner_expanded_tv<1>(*params.outputGradDesc);
                auto inputgrad_tv     = get_inner_expanded_tv<2>(*params.inputGradDesc);
                auto diagonal_tv = miopen::diag::getDiagonal(inputgrad_tv, params.diagonal, 0, 1);

                kernel(params.outputGrad, params.inputGrad, outgrad_numel, outgrad_tv, diagonal_tv);
            };
        };
    }

    return result;
}

} // namespace diag

} // namespace solver

} // namespace miopen
