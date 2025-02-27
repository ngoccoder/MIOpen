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

#include "miopen/mlo_internal.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/outer.hpp>
#include <miopen/outer/invoke_params.hpp>
#include <miopen/outer/solvers.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace outer {

bool IsImprovementOverROCm(const miopen::outer::FwdProblemDescription& problem)
{
    auto dtype = problem.GetX1Desc().GetType();
    auto ydims = problem.GetYDesc().GetLengths();

    if((ydims[0] <= 512) || (2048 < ydims[0] && ydims[1] <= 128) ||
       ((ydims[0] <= 2048 && ydims[1] <= 2048) &&
        (dtype == miopenHalf || dtype == miopenBFloat16)) ||
       ((ydims[0] <= 2048 && ydims[1] <= 512) && dtype == miopenFloat))
        return true;
    else
        return false;
}

bool OuterForward::IsApplicable(const ExecutionContext& /*context*/,
                                const miopen::outer::FwdProblemDescription& problem) const
{
    if(!(problem.GetX1Desc().GetType() == miopenFloat ||
         problem.GetX1Desc().GetType() == miopenHalf ||
         problem.GetX1Desc().GetType() == miopenBFloat16))
        return false;

    if(!IsImprovementOverROCm(problem))
        return false;

    return true;
}

ConvSolution OuterForward::GetSolution(const ExecutionContext& /*context*/,
                                       const miopen::outer::FwdProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype    = problem.GetX1Desc().GetType();
    auto io_dtype = miopen::GetDataType(dtype);
    auto y_numel  = problem.GetYDesc().GetElementSize();

    size_t xlocalsize = LOCAL_SIZE;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;
    size_t xgridsize  = AlignUp(y_numel, xlocalsize);
    size_t ygridsize  = 1;
    size_t zgridsize  = 1;

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenOuter.cpp";
    kernel.kernel_name = "OuterForward";

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

    result.invoker_factory = [y_numel](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::outer::FwdInvokeParams>();

            auto y_tv = get_inner_expanded_tv<2>(miopen::deref(params.yDesc));

            kernel(params.x1, params.x2, params.y, y_numel, y_tv);
        };
    };

    result.construction_params.push_back(kernel);

    return result;
}

} // namespace outer

} // namespace solver

} // namespace miopen
