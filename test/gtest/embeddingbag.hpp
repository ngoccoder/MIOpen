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

#include "cpu_embeddingbag.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "random.hpp"

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include <miopen/allocator.hpp>
#include <miopen/embeddingbag.hpp>
#include <miopen/miopen.h>

struct EmbeddingBagTestCase
{
    std::vector<size_t> input_dim;
    std::vector<size_t> weight_dim;
    std::vector<size_t> offsets;
    bool use_per_sample_weights;
    miopenEmbeddingBagMode_t mode;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const EmbeddingBagTestCase& tc)
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
        os << "Offsets dims: ";
        for(auto dim_sz : tc.offsets)
        {
            os << dim_sz << " ";
        }
        return os << " use_per_sample_weights: " << tc.use_per_sample_weights
                  << " mode: " << tc.mode << " contiguous: " << tc.isContiguous;
    }

    EmbeddingBagTestCase() {}

    EmbeddingBagTestCase(std::vector<size_t> input_dims_,
                         std::vector<size_t> weight_dims_,
                         std::vector<size_t> offsets_,
                         bool use_per_sample_weights_,
                         miopenEmbeddingBagMode_t mode_,
                         bool cont_)
        : input_dim(input_dims_),
          weight_dim(weight_dims_),
          offsets(offsets_),
          use_per_sample_weights(use_per_sample_weights_),
          mode(mode_),
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

inline std::vector<EmbeddingBagTestCase> GenFullTestCases()
{ // n c d h w dim
    // clang-format off
    return {
        {{16, 64}, {16, 16}, {}, false, MIOPEN_EMBEDDING_BAG_MEAN, true}, // cont no offsets (mode mean)
        {{16, 64}, {16, 16}, {}, false, MIOPEN_EMBEDDING_BAG_MEAN, false}, // noncont no offsets (mode mean)
        {{16, 64}, {16, 16}, {}, false, MIOPEN_EMBEDDING_BAG_SUM, true}, // cont no offsets (mode sum)
        {{16, 64}, {16, 16}, {}, false, MIOPEN_EMBEDDING_BAG_SUM, false}, // noncont no offsets (mode sum)
        {{16, 64}, {16, 16}, {}, true, MIOPEN_EMBEDDING_BAG_SUM, true}, // cont per sample weights (mode sum)
        {{16, 64}, {16, 16}, {}, true, MIOPEN_EMBEDDING_BAG_SUM, false}, // noncont per sample weights (mode sum)
        {{16, 64}, {16, 16}, {}, false, MIOPEN_EMBEDDING_BAG_MAX, true}, // cont no offsets (mode max)
        {{16, 64}, {16, 16}, {}, false, MIOPEN_EMBEDDING_BAG_MAX, false}, // noncont no offsets (mode max)
        {{40, 40}, {1024, 1024}, {}, false, MIOPEN_EMBEDDING_BAG_SUM, true}, // large cont no offsets (mode sum)
        {{40, 40}, {1024, 1024}, {}, false, MIOPEN_EMBEDDING_BAG_SUM, false}, // large noncont no offsets (mode sum)
        {{40, 40}, {1024, 1024}, {}, true, MIOPEN_EMBEDDING_BAG_SUM, true}, // large cont per sample weights (mode sum)
        {{40, 40}, {1024, 1024}, {}, true, MIOPEN_EMBEDDING_BAG_SUM, false}, // large noncont per sample weights (mode sum)
        {{40, 40}, {1024, 1024}, {}, false, MIOPEN_EMBEDDING_BAG_MEAN, true}, // large cont no offsets (mode mean)
    };
    // clang-format on
}

template <typename T>
struct EmbeddingBagFwdTest : public ::testing::TestWithParam<EmbeddingBagTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle       = get_handle();
        embeddingbag_config = GetParam();
        auto gen_value      = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto weight_dims   = embeddingbag_config.weight_dim;
        auto gen_value_int = [&weight_dims](auto...) {
            return prng::gen_0_to_B(static_cast<int64_t>(weight_dims[0] - 1));
        };

        mode = embeddingbag_config.mode;

        auto in_dims    = embeddingbag_config.input_dim;
        auto in_strides = embeddingbag_config.ComputeStrides(in_dims);
        input           = tensor<int64_t>{in_dims, in_strides}.generate(gen_value_int);

        weight = tensor<T>{weight_dims}.generate(gen_value);

        auto offsets_dims = embeddingbag_config.offsets;
        if(!offsets_dims.empty())
        {
            offsets = tensor<int64_t>{offsets_dims};
            for(int i = 0; i < offsets.GetSize(); i++)
            {
                if(i == 0)
                {
                    offsets[i] = prng::gen_A_to_B(static_cast<int64_t>(0),
                                                  static_cast<int64_t>(weight_dims[0] - 1));
                }
                else
                {
                    offsets[i] = prng::gen_A_to_B(static_cast<int64_t>(offsets[i - 1]),
                                                  static_cast<int64_t>(weight_dims[0] - 1));
                }
            }
        }

        if(embeddingbag_config.use_per_sample_weights)
        {
            per_sample_weights = tensor<T>{in_dims}.generate(gen_value);
        }

        std::vector<size_t> out_dims;
        if(in_dims.size() == 2)
        {
            out_dims.push_back(in_dims[0]);
        }
        else if(in_dims.size() == 1)
        {
            out_dims.push_back(offsets_dims[0]);
        }
        out_dims.push_back(weight_dims[1]);
        auto out_strides = embeddingbag_config.ComputeStrides(out_dims);
        output           = tensor<T>{out_dims, out_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims, out_strides};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        weight_dev = handle.Write(weight.data);
        output_dev = handle.Write(output.data);

        if(!offsets.data.empty())
        {
            offsets_dev = handle.Write(offsets.data);
        }
        if(!per_sample_weights.data.empty())
        {
            per_sample_weights_dev = handle.Write(per_sample_weights.data);
        }
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_embeddingbag_forward<T>(input, weight, offsets, per_sample_weights, ref_output, mode);
        status = miopen::embeddingbag::EmbeddingBagForward(handle,
                                                           input.desc,
                                                           input_dev.get(),
                                                           weight.desc,
                                                           weight_dev.get(),
                                                           offsets.desc,
                                                           offsets_dev.get(),
                                                           per_sample_weights.desc,
                                                           per_sample_weights_dev.get(),
                                                           output.desc,
                                                           output_dev.get(),
                                                           mode);

        ASSERT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        EXPECT_EQ(miopen::range_distance(output), miopen::range_distance(ref_output));
        auto error = miopen::rms_range(ref_output, output);

        EXPECT_LT(error, threshold * 10);
    }

    EmbeddingBagTestCase embeddingbag_config;

    tensor<int64_t> input;
    tensor<T> weight;
    tensor<int64_t> offsets;
    tensor<T> per_sample_weights;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr offsets_dev;
    miopen::Allocator::ManageDataPtr per_sample_weights_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    miopenEmbeddingBagMode_t mode;
};
