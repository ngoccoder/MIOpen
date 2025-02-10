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

#include "cpu_cosinesimilarity.hpp"
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
#include <miopen/cosinesimilarity.hpp>
#include <miopen/miopen.h>

struct CosineSimilarityTestCase
{
    std::vector<size_t> input1_dim;
    std::vector<size_t> input2_dim;
    uint32_t dim;
    float eps;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const CosineSimilarityTestCase& tc)
    {
        os << "Input1 dims: ";
        for(auto dim_sz : tc.input1_dim)
        {
            os << dim_sz << " ";
        }
        os << "Input2 dims: ";
        for(auto dim_sz : tc.input2_dim)
        {
            os << dim_sz << " ";
        }
        return os << " contiguous: " << tc.isContiguous << " dim: " << tc.dim << " eps: " << tc.eps;
    }

    CosineSimilarityTestCase() {}

    CosineSimilarityTestCase(std::vector<size_t> input1_dims_,
                             std::vector<size_t> input2_dims_,
                             uint32_t dim_,
                             float eps_,
                             bool cont_)
        : input1_dim(input1_dims_),
          input2_dim(input2_dims_),
          dim(dim_),
          eps(eps_),
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

inline std::vector<CosineSimilarityTestCase> GenFullTestCases()
{ // n c d h w dim
    // clang-format off
    return {
        {{256, 128, 256}, {256, 128, 256}, 0, 1e-8, true},
        {{256, 128, 256}, {256, 128, 256}, 0, 1e-8, false},
        {{32, 320, 64}, {32, 320, 64}, 0, 1e-8, true},
        {{32, 320, 64}, {32, 320, 64}, 0, 1e-8, false},
        {{320, 32, 64}, {320, 32, 64}, 1, 1e-8, true},
        {{320, 32, 64}, {320, 32, 64}, 1, 1e-8, false},
        {{32, 1024, 1024}, {32, 1024, 1024}, 0, 1e-8, true},
        {{32, 1024, 1024}, {32, 1024, 1024}, 0, 1e-8, false},
        {{32, 1024, 1024}, {32, 1024, 1024}, 1, 1e-8, true},
        {{32, 1024, 1024}, {32, 1024, 1024}, 1, 1e-8, false},
    };
    // clang-format on
}

template <typename T>
struct CosineSimilarityFwdTest : public ::testing::TestWithParam<CosineSimilarityTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        config          = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 99); };

        auto in1_dims    = config.input1_dim;
        auto in1_strides = config.ComputeStrides(in1_dims);
        input1           = tensor<T>{in1_dims, in1_strides}.generate(gen_value1);

        auto in2_dims    = config.input2_dim;
        auto in2_strides = config.ComputeStrides(in2_dims);
        input2           = tensor<T>{in2_dims, in2_strides}.generate(gen_value2);

        std::vector<size_t> out_dims;
        for(int i = 0; i < in1_dims.size(); i++)
        {
            if(i != config.dim)
            {
                out_dims.push_back(max(in1_dims[i], in2_dims[i]));
            }
        }
        auto out_strides = config.ComputeStrides(out_dims);
        output           = tensor<T>{out_dims, out_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims, out_strides};
        std::fill(ref_output.begin(), ref_output.end(), static_cast<T>(0));

        input1_dev = handle.Write(input1.data);
        input2_dev = handle.Write(input2.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_cosinesimilarity_forward(input1, input2, ref_output, config.dim, config.eps);
        status = miopen::cosinesimilarity::CosineSimilarityForward(handle,
                                                                   input1.desc,
                                                                   input1_dev.get(),
                                                                   input2.desc,
                                                                   input2_dev.get(),
                                                                   output.desc,
                                                                   output_dev.get(),
                                                                   config.dim,
                                                                   config.eps);

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
        EXPECT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        auto error = miopen::rms_range(ref_output, output);

        EXPECT_LT(error, threshold * 10);
    }

    CosineSimilarityTestCase config;

    tensor<T> input1;
    tensor<T> input2;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input1_dev;
    miopen::Allocator::ManageDataPtr input2_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    uint32_t dim;
    float eps;
};

template <typename T>
struct CosineSimilarityBwdTest : public ::testing::TestWithParam<CosineSimilarityTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        config          = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 99); };
        auto gen_value3 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 101); };

        auto in1_dims    = config.input1_dim;
        auto in1_strides = config.ComputeStrides(in1_dims);
        input1           = tensor<T>{in1_dims, in1_strides}.generate(gen_value1);

        auto in2_dims    = config.input2_dim;
        auto in2_strides = config.ComputeStrides(in2_dims);
        input2           = tensor<T>{in2_dims, in2_strides}.generate(gen_value2);

        std::vector<size_t> out_dims;
        for(int i = 0; i < in1_dims.size(); i++)
        {
            if(i != config.dim)
            {
                out_dims.push_back(max(in1_dims[i], in2_dims[i]));
            }
        }
        auto out_strides = config.ComputeStrides(out_dims);
        outputGrad       = tensor<T>{out_dims, out_strides}.generate(gen_value3);

        input1Grad = tensor<T>{in1_dims, in1_strides};
        input2Grad = tensor<T>{in2_dims, in2_strides};
        std::fill(input1Grad.begin(), input1Grad.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(input2Grad.begin(), input2Grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input1Grad = tensor<T>{in1_dims, in1_strides};
        std::fill(ref_input1Grad.begin(), ref_input1Grad.end(), static_cast<T>(0));
        ref_input2Grad = tensor<T>{in2_dims, in2_strides};
        std::fill(ref_input2Grad.begin(), ref_input2Grad.end(), static_cast<T>(0));

        input1_dev     = handle.Write(input1.data);
        input2_dev     = handle.Write(input2.data);
        outputGrad_dev = handle.Write(outputGrad.data);
        input1Grad_dev = handle.Write(input1Grad.data);
        input2Grad_dev = handle.Write(input2Grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_cosinesimilarity_backward(
            input1, input2, outputGrad, ref_input1Grad, ref_input2Grad, config.dim, config.eps);
        status = miopen::cosinesimilarity::CosineSimilarityBackward(handle,
                                                                    input1.desc,
                                                                    input1_dev.get(),
                                                                    input2.desc,
                                                                    input2_dev.get(),
                                                                    outputGrad.desc,
                                                                    outputGrad_dev.get(),
                                                                    input1Grad.desc,
                                                                    input1Grad_dev.get(),
                                                                    input2Grad.desc,
                                                                    input2Grad_dev.get(),
                                                                    config.dim,
                                                                    config.eps);

        ASSERT_EQ(status, miopenStatusSuccess);

        input1Grad.data = handle.Read<T>(input1Grad_dev, input1Grad.data.size());
        input2Grad.data = handle.Read<T>(input2Grad_dev, input2Grad.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        auto error1      = miopen::rms_range(ref_input1Grad, input1Grad);
        auto error2      = miopen::rms_range(ref_input2Grad, input2Grad);

        EXPECT_EQ(miopen::range_distance(ref_input1Grad), miopen::range_distance(input1Grad));
        EXPECT_EQ(miopen::range_distance(ref_input2Grad), miopen::range_distance(input2Grad));

        EXPECT_LT(error1, threshold * 10);
        EXPECT_LT(error2, threshold * 10);
    }

    CosineSimilarityTestCase config;

    tensor<T> input1;
    tensor<T> input2;
    tensor<T> outputGrad;
    tensor<T> input1Grad;
    tensor<T> input2Grad;

    tensor<T> ref_input1Grad;
    tensor<T> ref_input2Grad;

    miopen::Allocator::ManageDataPtr input1_dev;
    miopen::Allocator::ManageDataPtr input2_dev;
    miopen::Allocator::ManageDataPtr outputGrad_dev;
    miopen::Allocator::ManageDataPtr input1Grad_dev;
    miopen::Allocator::ManageDataPtr input2Grad_dev;

    uint32_t dim;
    float eps;
};
