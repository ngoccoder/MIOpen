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

#include <miopen/datatype.hpp>
#include <miopen/embeddingbag/invoke_params.hpp>
#include <miopen/embeddingbag/problem_description.hpp>
#include <miopen/embeddingbag/solvers.hpp>
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

namespace embeddingbag {

bool EmbeddingBagForward::IsApplicable(
    const ExecutionContext& /*context*/,
    const miopen::embeddingbag::FwdProblemDescription& problem) const
{
    if(!(problem.GetWeightDesc().GetType() == miopenFloat ||
         problem.GetWeightDesc().GetType() == miopenHalf ||
         problem.GetWeightDesc().GetType() == miopenBFloat16))
    {
        return false;
    }

    if(problem.GetOffsetsDesc().IsDefined())
    {
        return false;
    }

    return true;
}

ConvSolution
EmbeddingBagForward::GetSolution(const ExecutionContext& /*context*/,
                                 const miopen::embeddingbag::FwdProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetWeightDesc().GetType();
    auto io_dtype     = miopen::GetDataType(dtype);
    auto output_numel = problem.GetOutputDesc().GetElementSize();
    auto mode         = problem.GetMode();

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenEmbeddingBag.cpp";
    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(output_numel, xlocalsize);

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);
    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    kernel.kernel_name =
        (mode == MIOPEN_EMBEDDING_BAG_MAX) ? "EmbeddingBagMaxForward" : "EmbeddingBagForward";

    result.construction_params.push_back(kernel);

    result.invoker_factory = [mode](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) kernel = handle_.Run(kernels.front());
            decltype(auto) params = raw_params.CastTo<miopen::embeddingbag::FwdInvokeParams>();

            tensor_view_t<2> input_tv  = get_inner_expanded_tv<2>(deref(params.inputDesc));
            tensor_view_t<2> weight_tv = get_inner_expanded_tv<2>(deref(params.weightDesc));
            tensor_view_t<2> output_tv = get_inner_expanded_tv<2>(deref(params.outputDesc));
            tensor_view_t<2> per_sample_weights_tv =
                get_inner_expanded_tv<2>(deref(params.perSampleWeightDesc));
            if(mode == MIOPEN_EMBEDDING_BAG_MAX)
            {
                kernel(params.input, params.weight, params.output, input_tv, weight_tv, output_tv);
            }
            else
            {
                kernel(params.input,
                       params.weight,
                       params.output,
                       params.perSampleWeight,
                       static_cast<int32_t>(mode),
                       input_tv,
                       weight_tv,
                       output_tv,
                       per_sample_weights_tv);
            }
        };
    };

    return result;
}

} // namespace embeddingbag

} // namespace solver

} // namespace miopen
