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
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/softmax/invoke_params.hpp>
#include <miopen/softmax/solvers.hpp>
#include <miopen/target_properties.hpp>

#include <cstddef>

#define LOCAL_SIZE 256
#define CHUNK_SIZE 64

namespace miopen {

namespace solver {

namespace softmax {

bool SoftmaxV3Backward::IsApplicable(const ExecutionContext& context,
                                     const miopen::softmax::ProblemDescription& problem) const
{
    std::ignore = context;

    if(!(problem.GetXDesc().GetType() == miopenFloat ||
         problem.GetXDesc().GetType() == miopenHalf ||
         problem.GetXDesc().GetType() == miopenBFloat16))
        return false;

    // if(problem.GetXDesc().GetLengths()[problem.GetDim()] <= 8)
    //    return false;

    if(!problem.IsAllContiguous())
        return false;

    // if(!(problem.IsAllStrideOne() || problem.GetDim() < problem.GetXDesc().GetNumDims() - 1))
    //    return false;

    return true;
}

ConvSolution
SoftmaxV3Backward::GetSolution(const ExecutionContext& context,
                               const miopen::softmax::ProblemDescription& problem) const
{
    std::ignore = context;

    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype          = problem.GetXDesc().GetType();
    auto io_dtype       = miopen::GetDataType(problem.GetXDesc().GetType());
    auto dim            = problem.GetDim();
    auto output_len     = problem.GetXDesc().GetLengths();
    auto reduce_size    = output_len[dim];
    uint64_t inner_size = 1;
    for(size_t i = dim + 1; i < output_len.size(); i++)
    {
        inner_size *= output_len[i];
    }
    uint64_t outer_size = 1;
    for(uint32_t i = 0; i < dim; i++)
    {
        outer_size *= output_len[i];
    }

    auto isAllStrideOne    = problem.IsAllStrideOne();
    auto isReduceSizeSmall = problem.IsReduceSizeSmall();

    auto kernel        = KernelInfo{};
    kernel.kernel_file = "MIOpenSoftmaxV3.cpp";

    size_t xlocalsize = 1;
    size_t xgridsize  = 1;
    size_t ylocalsize = 1;
    size_t ygridsize  = 1;
    size_t zlocalsize = 1;
    size_t zgridsize  = 1;

    if(isReduceSizeSmall)
    {
        kernel.kernel_name = "SoftmaxBwdSmallContiguous";
        xlocalsize         = LOCAL_SIZE;
        xgridsize          = outer_size * inner_size * xlocalsize;
    }
    else
    {
        if(isAllStrideOne)
        {
            kernel.kernel_name = "SoftmaxBwdStrideOneContiguous";
            xlocalsize         = LOCAL_SIZE;
            xgridsize          = outer_size * inner_size * xlocalsize;
        }
        else
        {
            kernel.kernel_name = "SoftmaxBwdDimIsNotLastContiguous";
            xlocalsize         = LOCAL_SIZE;
            xgridsize = outer_size * ((inner_size + CHUNK_SIZE - 1) / CHUNK_SIZE) * LOCAL_SIZE;
        }
    }

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

    result.invoker_factory =
        [isAllStrideOne, isReduceSizeSmall, reduce_size, inner_size, outer_size](
            const std::vector<Kernel>& kernels) {
            return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
                decltype(auto) kernel = handle_.Run(kernels.front());
                decltype(auto) params = raw_params.CastTo<miopen::softmax::InvokeParams>();

                if(isReduceSizeSmall)
                {
                    kernel(params.backward_y,
                           params.dy,
                           params.dx,
                           reduce_size,
                           inner_size,
                           outer_size,
                           params.mode);
                }
                else
                {
                    if(isAllStrideOne)
                    {
                        kernel(params.backward_y, params.dy, params.dx, reduce_size, params.mode);
                    }
                    else
                    {
                        kernel(params.backward_y,
                               params.dy,
                               params.dx,
                               reduce_size,
                               inner_size,
                               params.mode);
                    }
                }
            };
        };

    return result;
}

} // namespace softmax

} // namespace solver

} // namespace miopen
