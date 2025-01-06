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
#pragma once

#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

#include "tensor_holder.hpp"
#include "tensor_view.hpp"

#include <cstddef>

template <class T>
void cpu_embeddingbag_forward(const tensor<int64_t>& input,
                              const tensor<T>& weight,
                              const tensor<int64_t>& offsets,
                              const tensor<T>& per_sample_weights,
                              tensor<T>& output,
                              miopenEmbeddingBagMode_t mode)
{
    auto input_len             = input.desc.GetLengths();
    tensor_view_t<2> weight_tv = get_inner_expanded_tv<2>(weight.desc);
    int64_t num_embeddings     = weight_tv.size[0];
    tensor_view_t<2> output_tv = get_inner_expanded_tv<2>(output.desc);
    auto output_numel          = output.desc.GetElementSize();

    if(input_len.size() == 1) // with offsets
    {
        tensor_view_t<1> input_tv   = get_inner_expanded_tv<1>(input.desc);
        tensor_view_t<1> offsets_tv = get_inner_expanded_tv<1>(offsets.desc);
        for(size_t o = 0; o < output_numel; o++)
        {
            tensor_layout_t<2> output_layout(output_tv, o);
            size_t bag         = output_layout.layout[0];
            size_t feature_dim = output_layout.layout[1];

            size_t input_start = offsets[bag];
            size_t input_end = (bag + 1 < offsets_tv.size[0]) ? offsets[bag + 1] : input_tv.size[0];
            auto divisor     = input_end - input_start;
            double res =
                (mode == MIOPEN_EMBEDDING_BAG_MAX) ? std::numeric_limits<double>::min() : 0;

            for(auto i = input_start; i < input_end; i++)
            {
                int64_t embedding_idx = input[i];

                if(embedding_idx >= 0 && embedding_idx < num_embeddings)
                {
                    double w = static_cast<double>(
                        weight[weight_tv.get_tensor_view_idx({embedding_idx, feature_dim})]);
                    if(mode == MIOPEN_EMBEDDING_BAG_MAX)
                    {
                        res = std::max(res, w);
                    }
                    else
                    {
                        T scale = per_sample_weights.data.data() ? per_sample_weights[i]
                                                                 : static_cast<T>(1);
                        res += w * static_cast<double>(scale);
                    }
                }
            }

            output[output_tv.get_tensor_view_idx(output_layout)] =
                (mode == MIOPEN_EMBEDDING_BAG_MEAN) ? (divisor ? (res / divisor) : 0) : res;
        }
    }
    else if(input_len.size() == 2) // no offsets
    {
        tensor_view_t<2> input_tv              = get_inner_expanded_tv<2>(input.desc);
        tensor_view_t<2> per_sample_weights_tv = get_inner_expanded_tv<2>(per_sample_weights.desc);
        for(size_t o = 0; o < output_numel; o++)
        {
            tensor_layout_t<2> output_layout(output_tv, o);
            double res =
                (mode == MIOPEN_EMBEDDING_BAG_MAX) ? std::numeric_limits<double>::min() : 0;

            for(size_t i = 0; i < input_tv.size[1]; i++)
            {
                int64_t embedding_idx =
                    input[input_tv.get_tensor_view_idx({output_layout.layout[0], i})];
                if(embedding_idx >= 0 && embedding_idx < num_embeddings)
                {
                    double w = static_cast<double>(weight[weight_tv.get_tensor_view_idx(
                        {embedding_idx, output_layout.layout[1]})]);
                    if(mode == MIOPEN_EMBEDDING_BAG_MAX)
                    {
                        res = std::max(res, w);
                    }
                    else
                    {
                        T scale =
                            per_sample_weights.data.data()
                                ? per_sample_weights[per_sample_weights_tv.get_tensor_view_idx(
                                      {output_layout.layout[0], i})]
                                : static_cast<T>(1);
                        res += w * static_cast<double>(scale);
                    }
                }
            }
            output[output_tv.get_tensor_view_idx(output_layout)] =
                (mode == MIOPEN_EMBEDDING_BAG_MEAN) ? res / input_tv.size[1] : res;
        }
    }
}
