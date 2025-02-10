/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/cosinesimilarity/invoke_params.hpp>
#include <miopen/cosinesimilarity/problem_description.hpp>
#include <miopen/cosinesimilarity/solvers.hpp>
#include <miopen/datatype.hpp>
#include <miopen/errors.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <cstddef>
#include <vector>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace cosinesimilarity {

bool CosineSimilarityBackward::IsImprovementOverROCm(
    const ExecutionContext& /*context*/,
    const miopen::cosinesimilarity::BwdProblemDescription& problem) const
{
    if(problem.GetOutputGradDesc().GetElementSize() < 20000)
    {
        return false;
    }

    return true;
}

bool CosineSimilarityBackward::IsApplicable(
    const ExecutionContext& context,
    const miopen::cosinesimilarity::BwdProblemDescription& problem) const
{
    if(!(problem.GetInput1Desc().GetType() == miopenFloat ||
         problem.GetInput1Desc().GetType() == miopenHalf ||
         problem.GetInput1Desc().GetType() == miopenBFloat16))
    {
        return false;
    }

    if(!IsImprovementOverROCm(context, problem))
    {
        return false;
    }

    return true;
}

ConvSolution CosineSimilarityBackward::GetSolution(
    const ExecutionContext& context,
    const miopen::cosinesimilarity::BwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetInput1Desc().GetType();
    auto io_dtype     = miopen::GetDataType(problem.GetInput1Desc().GetType());
    auto output_numel = problem.GetOutputGradDesc().GetElementSize();

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(output_numel, xlocalsize);
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    auto kernel = KernelInfo{};

    kernel.kernel_file = "MIOpenCosineSimilarity.cpp";
    kernel.kernel_name = "CosineSimilarityBackward";

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

    result.invoker_factory = [output_numel](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::cosinesimilarity::BwdInvokeParams>();

            tensor_view_t<5> input1_tv = get_inner_expanded_tv<5>(miopen::deref(params.input1Desc));
            tensor_view_t<5> input2_tv = get_inner_expanded_tv<5>(miopen::deref(params.input2Desc));
            tensor_view_t<5> input1_grad_tv =
                get_inner_expanded_tv<5>(miopen::deref(params.input1GradDesc));
            tensor_view_t<5> input2_grad_tv =
                get_inner_expanded_tv<5>(miopen::deref(params.input2GradDesc));
            tensor_view_t<4> output_grad_tv_4d =
                get_inner_expanded_tv<4>(miopen::deref(params.outputGradDesc));
            auto output_grad_tv = output_grad_tv_4d.unsqueeze(params.dim);

            kernel(params.input1,
                   params.input2,
                   params.outputGrad,
                   params.input1Grad,
                   params.input2Grad,
                   output_numel,
                   params.dim,
                   params.eps,
                   input1_tv,
                   input2_tv,
                   output_grad_tv,
                   input1_grad_tv,
                   input2_grad_tv);
        };
    };

    return result;
}

} // namespace cosinesimilarity

} // namespace solver

} // namespace miopen
