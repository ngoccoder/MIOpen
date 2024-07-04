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

#include "miopen/kernel_info.hpp"
#include "miopen/mlo_internal.hpp"
#include "miopen/tensor.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <cstddef>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/diagonal/diag/invoke_params.hpp>
#include <miopen/diagonal/solvers.hpp>
#include <miopen/diagonal.hpp>
#include <miopen/target_properties.hpp>

#define LOCAL_SIZE 256

namespace miopen {

namespace solver {

namespace diagonal {

tensor_view_t<5>
getDiagonal(const TensorDescriptor& tensor, int64_t offset, int64_t dim1, int64_t dim2)
{
    if(dim1 == dim2)
    {
        MIOPEN_THROW(miopenStatusInternalError, "Diagonal dimensions can not be identical");
    }

    int64_t diag_size;
    auto lens     = tensor.GetLengths();
    size_t dimNum = lens.size();
    auto strides  = tensor.GetStrides();
    if(offset >= 0)
    {
        diag_size = std::max<int64_t>(std::min(lens[dim1], lens[dim2] - offset), 0);
    }
    else
    {
        diag_size = std::max<int64_t>(std::min(lens[dim1] + offset, lens[dim2]), 0);
    }

    uint64_t new_offset = 0;
    if(diag_size == 0)
    {
        // skip
    }
    else if(offset >= 0)
    {
        new_offset += offset * strides[dim2];
    }
    else
    {
        new_offset -= offset * strides[dim1];
    }

    tensor_view_t<5> res;
    res.offset = new_offset;

    int curIdx    = 0;
    int curNewIdx = 0;
    while(curNewIdx < dimNum - 2)
    {
        if(curIdx == dim1 || curIdx == dim2)
        {
            curIdx++;
        }
        else
        {
            res.size[curNewIdx]   = lens[curIdx];
            res.stride[curNewIdx] = strides[curIdx];
            curNewIdx++;
            curIdx++;
        }
    }
    res.size[dimNum - 2]   = diag_size;
    res.stride[dimNum - 2] = strides[dim1] + strides[dim2];

    for(int i = dimNum - 1; i < 5; ++i)
    {
        res.stride[i] = (i == 0 ? 1 : res.stride[i - 1]);
        res.size[i]   = 1;
    }

    return res;
}

namespace diag {

bool DiagForward::IsApplicable(const ExecutionContext& context,
                               const miopen::diagonal::diag::FwdProblemDescription& problem) const
{
    std::ignore    = context;
    auto inputDims = problem.GetInputDesc().GetLengths();

    if(!problem.IsSameType())
        return false;
    if(!problem.IsAllPacked())
        return false;
    return true;
}

ConvSolution
DiagForward::GetSolution(const ExecutionContext& context,
                         const miopen::diagonal::diag::FwdProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype         = problem.GetInputDesc().GetType();
    auto input_dtype   = miopen::GetDataType(problem.GetInputDesc().GetType());
    auto output_dtype  = miopen::GetDataType(problem.GetOutputDesc().GetType());
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

    if(problem.GetInputDesc().GetLengths().size() == 1)
    {
        auto input_numel = problem.GetInputDesc().GetElementSize();

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(input_numel, xlocalsize);
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
                decltype(auto) params =
                    raw_params.CastTo<miopen::diagonal::diag::FwdInvokeParams>();
                auto input_numel = params.inputDesc->GetElementSize();
                auto output_tv   = get_inner_expanded_tv<2>(*params.outputDesc);
                long offset      = (params.diagonal >= 0 ? params.diagonal * output_tv.stride[1]
                                                         : -params.diagonal * output_tv.stride[0]);

                kernel(params.input, params.output, input_numel, offset, output_tv);
            };
        };
    }
    else if(problem.GetInputDesc().GetLengths().size() == 2)
    {
        int64_t output_numel = problem.GetOutputDesc().GetElementSize();

        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(output_numel, xlocalsize);
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
                decltype(auto) params =
                    raw_params.CastTo<miopen::diagonal::diag::FwdInvokeParams>();
                auto output_numel = params.outputDesc->GetElementSize();
                auto input_tv     = get_inner_expanded_tv<2>(*params.inputDesc);
                long offset       = (params.diagonal >= 0 ? params.diagonal * input_tv.stride[1]
                                                          : -params.diagonal * input_tv.stride[0]);

                kernel(params.input, params.output, output_numel, offset, input_tv);
            };
        };
    }

    return result;
}

} // namespace diag

} // namespace diagonal

} // namespace solver

} // namespace miopen
