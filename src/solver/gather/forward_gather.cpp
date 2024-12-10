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
#include <miopen/errors.hpp>
#include <miopen/gather.hpp>
#include <miopen/gather/invoke_params.hpp>
#include <miopen/gather/problem_description.hpp>
#include <miopen/gather/solvers.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/tensor_view_utils.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace gather {

bool GatherForward::IsApplicable(const ExecutionContext& /*context*/,
                                 const miopen::gather::FwdProblemDescription& problem) const
{
    auto inputDesc = problem.GetInputDesc();
    if(inputDesc.GetType() != miopenFloat && inputDesc.GetType() != miopenHalf &&
       inputDesc.GetType() != miopenBFloat16)
    {
        return false;
    }

    if(problem.GetGatherDesc().getMode() != MIOPEN_GATHER)
    {
        return false;
    }
    return true;
}

ConvSolution GatherForward::GetSolution(const ExecutionContext& context,
                                        const miopen::gather::FwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetInputDesc().GetType();
    auto in_out_dtype = miopen::GetDataType(dtype);
    auto indices_type = miopen::GetDataType(problem.GetIndicesDesc().GetType());

    auto output_numel = problem.GetOutputDesc().GetElementSize();

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IO_TYPE", in_out_dtype == "bfloat16" ? "ushort" : in_out_dtype},
        {"INDEX_TYPE", indices_type == "int64" ? "size_t" : indices_type},
    };

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(output_numel, xlocalsize);

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenGather.cpp";

    kernel.kernel_name  = "GatherForward";
    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);
    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    result.construction_params.push_back(kernel);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::gather::FwdInvokeParams>();

            auto input_tv   = get_inner_expanded_tv<5>(deref(params.inputDesc));
            auto indices_tv = get_inner_expanded_tv<5>(deref(params.indicesDesc));
            auto output_tv  = get_inner_expanded_tv<5>(deref(params.outputDesc));

            kernel(params.input,
                   params.indices,
                   params.output,
                   params.dim,
                   input_tv,
                   indices_tv,
                   output_tv);
        };
    };

    return result;
}

} // namespace gather

} // namespace solver

} // namespace miopen
