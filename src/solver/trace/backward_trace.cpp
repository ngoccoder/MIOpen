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

#include "miopen/errors.hpp"
#include <miopen/common.hpp>
#include <miopen/datatype.hpp>
#include <miopen/hipoc_kernel.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/trace/invoke_params.hpp>
#include <miopen/trace/problem_description.hpp>
#include <miopen/trace/solvers.hpp>
#include <miopen/miopen.h>
#include <miopen/mlo_internal.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <cstddef>
#include <vector>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace trace {

bool TraceBackward::IsApplicable(const ExecutionContext& /*context*/,
                                 const miopen::trace::BwdProblemDescription& problem) const
{
    if(!(problem.GetInputGradDesc().GetType() == miopenFloat ||
         problem.GetInputGradDesc().GetType() == miopenHalf ||
         problem.GetInputGradDesc().GetType() == miopenBFloat16))
    {
        return false;
    }

    return true;
}

ConvSolution TraceBackward::GetSolution(const ExecutionContext& /*context*/,
                                        const miopen::trace::BwdProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype            = problem.GetOutputGradDesc().GetType();
    auto io_dtype         = miopen::GetDataType(dtype);
    auto input_grad_len   = problem.GetInputGradDesc().GetLengths();
    size_t N              = input_grad_len[0];
    auto input_grad_numel = problem.GetInputGradDesc().GetElementSize();

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype}};

    /* Phase 1: Fill input grad with zeros */
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(input_grad_numel, xlocalsize);

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenFill.cpp";
        kernel.kernel_name = "FillZero";

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    /* Phase 2: Trace backward */
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(N, xlocalsize);

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenTrace.cpp";
        kernel.kernel_name = "TraceBackward";

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [input_grad_numel, N](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::trace::BwdInvokeParams>();

            float elapsed = 0.f;
            HipEventPtr start, stop;

            bool reset_profiling_state = false;
            if(handle_.IsProfilingEnabled())
            {
                reset_profiling_state = true;
                handle_.EnableProfiling(false);
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            /* Phase 1: Fill input grad with zeros. */
            {
                decltype(auto) kernel = handle_.Run(kernels.front());
                kernel(params.inputGrad, input_grad_numel);
            }

            /* Phase 2: Trace backward. */
            {
                auto output_grad_tv = get_inner_expanded_tv<1>(deref(params.outputGradDesc));
                auto input_grad_tv  = get_inner_expanded_tv<2>(deref(params.inputGradDesc));

                decltype(auto) kernel = handle_.Run(kernels.back());
                kernel(params.outputGrad, params.inputGrad, N, output_grad_tv, input_grad_tv);
            }

            if(reset_profiling_state)
            {
                handle_.EnableProfiling(true);
            }

            if(handle_.IsProfilingEnabled())
            {
                hipEventRecord(stop.get(), handle_.GetStream());
                hipEventSynchronize(stop.get());
                hipEventElapsedTime(&elapsed, start.get(), stop.get());

                // Clean up
                hipEventDestroy(start.get());
                hipEventDestroy(stop.get());
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            }
        };
    };

    return result;
}

} // namespace trace

} // namespace solver

} // namespace miopen
