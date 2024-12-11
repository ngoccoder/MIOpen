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

#include <cstddef>

#include <miopen/datatype.hpp>
#include <miopen/gather.hpp>
#include <miopen/gather/invoke_params.hpp>
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

bool GatherNDBackward::IsApplicable(const ExecutionContext& /*context*/,
                                    const miopen::gather::BwdProblemDescription& problem) const
{
    auto paramGradDesc = problem.GetParamGradDesc();
    if(paramGradDesc.GetType() != miopenFloat && paramGradDesc.GetType() != miopenHalf &&
       paramGradDesc.GetType() != miopenBFloat16)
    {
        return false;
    }
    if(problem.GetGatherDesc().getMode() != MIOPEN_GATHER_ND)
    {
        return false;
    }
    return true;
}

ConvSolution
GatherNDBackward::GetSolution(const ExecutionContext& context,
                              const miopen::gather::BwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetParamGradDesc().GetType();
    auto in_out_dtype = miopen::GetDataType(dtype);
    auto indices_type = miopen::GetDataType(problem.GetIndicesDesc().GetType());

    auto param_grad_len     = problem.GetParamGradDesc().GetLengths();
    auto outGrad_numel      = problem.GetOutputGradDesc().GetElementSize();
    auto paramGrad_numel    = problem.GetParamGradDesc().GetElementSize();
    auto param_grad_num_dim = problem.GetParamGradDesc().GetNumDims();
    auto indices_num_dim    = problem.GetIndicesDesc().GetNumDims();
    auto indices_len        = problem.GetIndicesDesc().GetLengths();
    auto slice_dim          = (indices_num_dim > 1) ? indices_len[indices_num_dim - 1] : 1;

    size_t slice_size = 1;
    for(size_t i = slice_dim; i < param_grad_num_dim; i++)
    {
        slice_size *= param_grad_len[i];
    }

    size_t num_indices = 1;
    for(size_t i = 0; i < indices_num_dim - 1; i++)
    {
        num_indices *= indices_len[i];
    }

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IO_TYPE", in_out_dtype == "bfloat16" ? "ushort" : in_out_dtype},
        {"INDEX_TYPE", indices_type == "int64" ? "size_t" : indices_type},
    };

    /* Phase 1: Fill param grad with zeros */
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(paramGrad_numel, xlocalsize);

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

    /* Phase 2: Gather backward */
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(outGrad_numel, xlocalsize);

        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenScatterND.cpp";
        kernel.kernel_name = "ScatterNDAddForward";

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [paramGrad_numel, num_indices, slice_size, slice_dim](
                                 const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::gather::BwdInvokeParams>();

            float elapsed = 0.0f;
            HipEventPtr start;
            HipEventPtr stop;

            bool reset_profiling_state = false;
            if(handle_.IsProfilingEnabled())
            {
                reset_profiling_state = true;
                handle_.EnableProfiling(false);
                start = miopen::make_hip_event();
                stop  = miopen::make_hip_event();
                hipEventRecord(start.get(), handle_.GetStream());
            }

            /* Phase 1: Fill param grad with zeros */
            {
                decltype(auto) kernel = handle_.Run(kernels.front());
                kernel(params.paramGrad, paramGrad_numel);
            }

            /* Phase 2: GatherND backward */
            {
                decltype(auto) kernel = handle_.Run(kernels.back());
                auto param_grad_tv = get_inner_expanded_tv<9>(miopen::deref(params.paramGradDesc));

                kernel(params.outputGrad,
                       params.indices,
                       params.paramGrad,
                       num_indices,
                       slice_size,
                       slice_dim,
                       param_grad_tv);
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

} // namespace gather

} // namespace solver

} // namespace miopen
