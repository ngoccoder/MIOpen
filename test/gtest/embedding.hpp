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

#include "cpu_embedding.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "random.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <unordered_map>
#include <vector>

#include <miopen/allocator.hpp>
#include <miopen/embedding.hpp>
#include <miopen/miopen.h>

struct EmbeddingTestCase
{
    std::vector<size_t> input_dim;
    std::vector<size_t> weight_dim;
    bool scale_grad_by_freq;
    int64_t padding_idx;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const EmbeddingTestCase& tc)
    {
        os << "Input dims: ";
        for(auto dim_sz : tc.input_dim)
        {
            os << dim_sz << " ";
        }
        os << "Weight dims: ";
        for(auto dim_sz : tc.weight_dim)
        {
            os << dim_sz << " ";
        }
        return os << " contiguous: " << tc.isContiguous
                  << " scale_grad_by_freq: " << tc.scale_grad_by_freq
                  << " padding_idx: " << tc.padding_idx;
    }

    EmbeddingTestCase() {}

    EmbeddingTestCase(std::vector<size_t> input_dims_,
                      std::vector<size_t> weight_dims_,
                      bool scale_grad_by_freq_,
                      int64_t padding_idx_,
                      bool cont_)
        : input_dim(input_dims_),
          weight_dim(weight_dims_),
          scale_grad_by_freq(scale_grad_by_freq_),
          padding_idx(padding_idx_),
          isContiguous(cont_)
    {
    }

    std::vector<size_t> ComputeStrides(const std::vector<size_t>& input_dim_) const
    {
        std::vector<size_t> inputDim = input_dim_;
        if(!isContiguous)
            std::swap(inputDim.front(), inputDim.back());
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!isContiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

inline std::vector<EmbeddingTestCase> GenFullTestCases()
{ // n c d h w dim
    // clang-format off
    return {
        {{16, 4, 4}, {16, 16}, false, 0, true}, // cont small weight grad
        {{16, 4, 4}, {16, 16}, false, 0, false}, // non-cont small weight grad
        {{16, 4, 4}, {16, 16}, true, 0, true}, // cont small weight grad (scale)
        {{16, 4, 4}, {16, 16}, true, 0, false}, // non-cont small weight grad (scale)
        {{32, 32, 64}, {64, 64}, false, 0, true}, // cont large weight grad
        {{32, 32, 64}, {64, 64}, false, 0, false}, // non-cont large weight grad
        {{32, 32, 64}, {64, 64}, true, 0, true}, // cont large weight grad (scale)
        {{32, 32, 64}, {64, 64}, true, 0, false}, // non-cont large weight grad (scale)
        {{64, 64, 128}, {1024, 1024}, false, 0, true}, // cont large weight grad
        {{64, 64, 128}, {1024, 1024}, false, 0, false}, // non-cont large weight grad
        {{64, 64, 128}, {1024, 1024}, true, 0, true}, // cont large weight grad (scale)
        {{64, 64, 128}, {1024, 1024}, true, 0, false}, // non-cont large weight grad (scale)
    };
    // clang-format on
}

template <typename T>
struct EmbeddingBwdTest : public ::testing::TestWithParam<EmbeddingTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle      = get_handle();
        embedding_config   = GetParam();
        auto gen_value     = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value_int = [](auto...) {
            return prng::gen_descreet_uniform_sign<int32_t>(1e-2, 100);
        };

        auto in_dims    = embedding_config.input_dim;
        auto in_strides = embedding_config.ComputeStrides(in_dims);
        input           = tensor<int64_t>{in_dims, in_strides}.generate(gen_value_int);

        auto weight_dims    = embedding_config.weight_dim;
        auto weight_strides = embedding_config.ComputeStrides(weight_dims);
        weightGrad          = tensor<T>{weight_dims, weight_strides};
        std::fill(weightGrad.begin(), weightGrad.end(), std::numeric_limits<T>::quiet_NaN());

        auto out_dims = in_dims;
        out_dims.push_back(weight_dims[1]);
        auto out_strides = embedding_config.ComputeStrides(out_dims);
        outputGrad       = tensor<T>{out_dims, out_strides}.generate(gen_value);

        ref_weightGrad = tensor<T>{weight_dims};
        std::fill(ref_weightGrad.begin(), ref_weightGrad.end(), static_cast<T>(0));

        input_dev      = handle.Write(input.data);
        outputGrad_dev = handle.Write(outputGrad.data);
        weightGrad_dev = handle.Write(weightGrad.data);

        if(embedding_config.scale_grad_by_freq)
        {
            indices_freq = std::vector<int32_t>(input.data.size(), 0);

            std::unordered_map<int64_t, int> counts;
            for(auto idx : input.data)
            {
                counts[idx]++;
            }
            for(size_t i = 0; i < input.data.size(); i++)
            {
                indices_freq[i] = counts[input.data[i]];
            }

            indices_freq_dev = handle.Write(indices_freq);
        }
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_embedding_backward<T>(
            input, outputGrad, ref_weightGrad, indices_freq, embedding_config.padding_idx);
        status = miopen::embedding::EmbeddingBackward(handle,
                                                      input.desc,
                                                      input_dev.get(),
                                                      outputGrad.desc,
                                                      outputGrad_dev.get(),
                                                      weightGrad.desc,
                                                      weightGrad_dev.get(),
                                                      indices_freq_dev.get(),
                                                      embedding_config.padding_idx);

        ASSERT_EQ(status, miopenStatusSuccess);

        weightGrad.data = handle.Read<T>(weightGrad_dev, weightGrad.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        EXPECT_EQ(miopen::range_distance(ref_weightGrad), miopen::range_distance(weightGrad));
        auto error = miopen::rms_range(ref_weightGrad, weightGrad);

        EXPECT_LT(error, threshold * 10) << "Error output beyond tolerance Error: " << error
                                         << ",  Tolerance: " << threshold * 10;
    }

    EmbeddingTestCase embedding_config;

    tensor<int64_t> input;
    tensor<T> outputGrad;
    tensor<T> weightGrad;

    tensor<T> ref_weightGrad;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr outputGrad_dev;
    miopen::Allocator::ManageDataPtr weightGrad_dev;

    std::vector<int32_t> indices_freq;
    miopen::Allocator::ManageDataPtr indices_freq_dev;
};
