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

#include "miopen/hipoc_kernel.hpp"
#include "miopen/invoke_params.hpp"
#include "miopen/kernel_info.hpp"
#include "miopen/miopen.h"
#include "miopen/mlo_internal.hpp"
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/gather/invoke_params.hpp>
#include <miopen/gather/solvers.hpp>
#include <miopen/gather.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace gather {

bool GatherV2Backward::IsApplicable(const ExecutionContext& /*context*/,
                                    const miopen::gather::BwdProblemDescription& problem) const
{
    return problem.GetGatherDesc().getMode() == MIOPEN_GATHER_V2;
}

ConvSolution
GatherV2Backward::GetSolution(const ExecutionContext& context,
                              const miopen::gather::BwdProblemDescription& problem) const
{
    std::ignore = context;
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetParamGradDesc().GetType();
    auto in_out_dtype = miopen::GetDataType(dtype);
    auto indices_type = miopen::GetDataType(problem.GetIndicesDesc().GetType());

    auto gatherDesc      = problem.GetGatherDesc();
    auto batch_dims      = gatherDesc.getBatchDims();
    auto paramGrad       = problem.GetParamGradDesc().GetLengths();
    auto outGrad_numel   = problem.GetOutputGradDesc().GetElementSize();
    auto indices_numel   = problem.GetIndicesDesc().GetElementSize();
    auto paramGrad_numel = problem.GetParamGradDesc().GetElementSize();
    auto dim             = gatherDesc.getDim();
    auto kernel          = KernelInfo{};

    const auto build_params = KernelBuildParameters{
        {"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
        {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
        {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
        {"IO_TYPE", in_out_dtype == "bfloat16" ? "ushort" : in_out_dtype},
        {"INDEX_TYPE", indices_type},
    };

    kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

    int64_t batch_size = 1;
    int64_t outer_size = 1;
    int64_t inner_size = 1;

    for(uint32_t i = 0; i < batch_dims; ++i)
    {
        batch_size *= paramGrad[i];
    }
    for(uint32_t i = batch_dims; i < dim; ++i)
    {
        outer_size *= paramGrad[i];
    }
    for(uint32_t i = dim + 1; i < paramGrad.size(); ++i)
    {
        inner_size *= paramGrad[i];
    }

    int64_t gather_dim_size = paramGrad[dim];

    kernel.kernel_file = "MIOpenGatherV2.cpp";

    size_t xlocalsize = LOCAL_SIZE;
    size_t xgridsize  = AlignUp(outGrad_numel, xlocalsize);

    kernel.l_wk.push_back(xlocalsize);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);
    kernel.g_wk.push_back(xgridsize);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);

    if(batch_dims > 0)
    {
        // Batched Gather Backward
        kernel.kernel_name = "BatchedGatherV2Backward";

        printf("batch size is %ld\n", batch_size);
        printf("outer_size is %ld\n", outer_size);
        printf("indices_numel / batch size is %ld\n", indices_numel / batch_size);
        printf("inner_size is %ld\n", inner_size);

        auto outputGrad_tv = miopen::gather::reshape<4>(
            problem.GetOutputGradDesc(),
            {batch_size, outer_size, indices_numel / batch_size, inner_size});

        result.construction_params.push_back(kernel);
        result.invoker_factory = [outputGrad_tv,
                                  paramGrad_numel,
                                  outGrad_numel,
                                  batch_size,
                                  outer_size,
                                  gather_dim_size,
                                  indices_numel,
                                  inner_size](const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::gather::BwdInvokeParams>();

                const bool is_batch_dims_zero = (batch_size == 1);
                const bool is_axis_zero       = (outer_size == 1);

                kernel(params.outputGrad,
                       params.indices,
                       params.paramGrad,
                       outputGrad_tv,
                       paramGrad_numel,
                       outer_size,
                       gather_dim_size,
                       indices_numel,
                       inner_size,
                       outGrad_numel,
                       is_axis_zero,
                       is_batch_dims_zero);
            };
        };
    }

    return result;
}

} // namespace gather

} // namespace solver

} // namespace miopen
