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

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/gatherv2/invoke_params.hpp>
#include <miopen/gatherv2/solvers.hpp>
#include <miopen/gatherv2.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace gatherv2 {

bool IsImprovementOverROCm(const miopen::gatherv2::BwdProblemDescription& problem) { return true; }

bool GatherV2Backward::IsApplicable(const ExecutionContext& /*context*/,
                                    const miopen::gatherv2::BwdProblemDescription& problem) const
{
    if(!IsImprovementOverROCm(problem))
    {
        return false;
    }

    return true;
}

ConvSolution
GatherV2Backward::GetSolution(const ExecutionContext& context,
                              const miopen::gatherv2::BwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype            = problem.GetParamGradDesc().GetType();
    auto in_out_dtype     = miopen::GetDataType(problem.GetParamGradDesc().GetType());
    auto batch_dims       = problem.GetBatchDims();
    auto paramGrad        = problem.GetParamGradDesc().GetLengths();
    auto outputGrad_numel = problem.GetOutputGradDesc().GetElementSize();
    auto axis             = problem.GetAxis();
    auto kernel           = KernelInfo{};

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IN_OUT_TYPE", in_out_dtype == "bfloat16" ? "ushort" : in_out_dtype}};

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    int64_t batch_size = 1;
    int64_t outer_size = 1;
    int64_t inner_size = 1;

    for(int i = 0; i < batch_dims; ++i)
    {
        batch_size *= paramGrad[i];
    }
    for(int i = batch_dims; i < axis; ++i)
    {
        outer_size *= paramGrad[i];
    }
    for(int i = axis + 1; i < paramGrad.size(); ++i)
    {
        inner_size *= paramGrad[i];
    }

    int64_t gather_dim_size = paramGrad[axis];
    const int64_t N         = problem.GetIndicesDesc().GetElementSize();

    if(batch_dims > 0)
    {
        auto paramGrad_tv = miopen::gatherv2::reshape<4>(
            problem.GetParamGradDesc(), {batch_size, outer_size, gather_dim_size, inner_size});
        auto outputGrad_tv = miopen::gatherv2::reshape<4>(
            problem.GetOutputGradDesc(), {batch_size, outer_size, N / batch_size, inner_size});
    }
    else
    {
        auto paramGrad_tv  = miopen::gatherv2::reshape<3>(problem.GetParamGradDesc(),
                                                         {outer_size, gather_dim_size, inner_size});
        auto outputGrad_tv = miopen::gatherv2::reshape<3>(problem.GetOutputGradDesc(),
                                                          {outer_size, N / batch_size, inner_size});
    }
    auto indices_tv = flatten(problem.GetIndicesDesc());

    kernel.kernel_file = "MIOpenGatherV2.cpp";

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(outputGrad_numel, xlocalsize);

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);
    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    if(paramGrad.size() == 4)
    {
        // Batched Gatther Backward
        kernel.kernel_name = "BatchedGatherV2Backward";

        result.construction_params.push_back(kernel);
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::gatherv2::BwdInvokeParams>();

                kernel(params.outputGrad, params.indices, params.paramGrad);
            };
        };
    }
    else
    {
        // Gather Backward
        kernel.kernel_name = "GatherV2Backward";

        result.construction_params.push_back(kernel);
        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::gatherv2::BwdInvokeParams>();

                kernel(params.outputGrad, params.indices, params.paramGrad);
            };
        };
    }

    return result;
}

} // namespace gatherv2

} // namespace solver

} // namespace miopen
