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
#include <miopen/embedding/invoke_params.hpp>
#include <miopen/embedding/problem_description.hpp>
#include <miopen/embedding/solvers.hpp>
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

namespace embedding {

bool EmbeddingBackward::IsApplicable(const ExecutionContext& /*context*/,
                                     const miopen::embedding::BwdProblemDescription& problem) const
{
    if(!(problem.GetWeightGradDesc().GetType() == miopenFloat ||
         problem.GetWeightGradDesc().GetType() == miopenHalf ||
         problem.GetWeightGradDesc().GetType() == miopenBFloat16))
    {
        return false;
    }

    return true;
}

ConvSolution
EmbeddingBackward::GetSolution(const ExecutionContext& /*context*/,
                               const miopen::embedding::BwdProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype                          = problem.GetWeightGradDesc().GetType();
    auto io_dtype                       = miopen::GetDataType(dtype);
    auto weight_len                     = problem.GetWeightGradDesc().GetLengths();
    auto num_embeddings                 = weight_len[0];
    auto embedding_dim                  = weight_len[1];
    auto weight_grad_numel              = problem.GetWeightGradDesc().GetElementSize();
    auto output_numel                   = problem.GetOutputGradDesc().GetElementSize();
    auto input_numel                    = problem.GetInputDesc().GetElementSize();
    bool use_embedding_bwd_contig       = problem.isAllContiguous();
    bool use_embedding_bwd_traverse_opt = (num_embeddings <= 32);
    int alpha                           = 0;

    const auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"IO_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype},
                              {"O_TYPE", io_dtype == "bfloat16" ? "ushort" : io_dtype}};

    /* Phase 1: Fill weight grad with zeros */
    {
        size_t xlocalsize = LOCAL_SIZE;
        size_t xgridsize  = AlignUp(weight_grad_numel, xlocalsize);

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

    /* Phase 2: Embedding backward */
    {
        auto kernel        = KernelInfo{};
        kernel.kernel_file = "MIOpenEmbedding.cpp";
        size_t xlocalsize  = LOCAL_SIZE;
        size_t xgridsize   = 1;

        if(use_embedding_bwd_traverse_opt)
        {
            kernel.kernel_name = use_embedding_bwd_contig
                                     ? "EmbeddingBackwardSmallNumEmbeddingsTraverseContiguous"
                                     : "EmbeddingBackwardSmallNumEmbeddingsTraverse";
            // magic number to give enough works while reduce contentions
            alpha     = 64;
            xgridsize = AlignUp(alpha * num_embeddings * embedding_dim, xlocalsize);
        }
        else
        {
            kernel.kernel_name = use_embedding_bwd_contig ? "EmbeddingBackwardContiguousAtomic"
                                                          : "EmbeddingBackwardAtomic";
            xgridsize          = AlignUp(output_numel, xlocalsize);
        }

        kernel.comp_options = build_params.GenerateFor(kbp::HIP{});

        kernel.l_wk.push_back(xlocalsize);
        kernel.l_wk.push_back(1);
        kernel.l_wk.push_back(1);
        kernel.g_wk.push_back(xgridsize);
        kernel.g_wk.push_back(1);
        kernel.g_wk.push_back(1);

        result.construction_params.push_back(kernel);
    }

    result.invoker_factory = [weight_grad_numel,
                              use_embedding_bwd_traverse_opt,
                              use_embedding_bwd_contig,
                              alpha,
                              embedding_dim,
                              num_embeddings,
                              input_numel](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::embedding::BwdInvokeParams>();

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

            /* Phase 1: Fill weight grad with zeros. */
            {
                decltype(auto) kernel = handle_.Run(kernels.front());
                kernel(params.weightGrad, weight_grad_numel);
            }

            /* Phase 2: Embedding backward. */
            {
                auto input_tv       = get_inner_expanded_tv<4>(deref(params.inputDesc));
                auto output_grad_tv = get_inner_expanded_tv<4>(deref(params.outputGradDesc));
                auto weight_grad_tv = get_inner_expanded_tv<2>(deref(params.weightGradDesc));

                decltype(auto) kernel = handle_.Run(kernels.back());
                if(use_embedding_bwd_traverse_opt)
                {
                    if(use_embedding_bwd_contig)
                    {
                        kernel(params.input,
                               params.outputGrad,
                               params.weightGrad,
                               params.indices_freq,
                               embedding_dim,
                               input_numel,
                               num_embeddings,
                               params.padding_idx,
                               alpha);
                    }
                    else
                    {
                        kernel(params.input,
                               params.outputGrad,
                               params.weightGrad,
                               params.indices_freq,
                               embedding_dim,
                               input_numel,
                               num_embeddings,
                               params.padding_idx,
                               alpha,
                               input_tv,
                               output_grad_tv,
                               weight_grad_tv);
                    }
                }
                else
                {
                    if(use_embedding_bwd_contig)
                    {
                        kernel(params.input,
                               params.outputGrad,
                               params.weightGrad,
                               params.indices_freq,
                               embedding_dim,
                               input_numel,
                               num_embeddings,
                               params.padding_idx);
                    }
                    else
                    {
                        kernel(params.input,
                               params.outputGrad,
                               params.weightGrad,
                               params.indices_freq,
                               embedding_dim,
                               input_numel,
                               num_embeddings,
                               params.padding_idx,
                               input_tv,
                               output_grad_tv,
                               weight_grad_tv);
                    }
                }
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

} // namespace embedding

} // namespace solver

} // namespace miopen
