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

#include "cpu_softmaxv3.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "random.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <limits>

#include <miopen/miopen.h>
#include <miopen/softmax.hpp>

struct SoftmaxTestCase
{
    std::vector<size_t> input_dim;
    uint32_t dim;
    miopenSoftmaxAlgorithm_t algorithm;

    friend std::ostream& operator<<(std::ostream& os, const SoftmaxTestCase& tc)
    {
        os << "Input dims: ";
        for(auto dim_sz : tc.input_dim)
        {
            os << dim_sz << " ";
        }
        return os << " dim: " << tc.dim << " algorithm: " << tc.algorithm;
    }

    SoftmaxTestCase() {}

    SoftmaxTestCase(std::vector<size_t> input_dims_, uint32_t dim_, miopenSoftmaxAlgorithm_t algo_)
        : input_dim(input_dims_), dim(dim_), algorithm(algo_)
    {
    }
};

inline std::vector<SoftmaxTestCase> GenFullTestCasesForward()
{
    return {
        {{16, 16, 16}, 0, MIOPEN_SOFTMAX_ACCURATE},     // small case (not last dim)
        {{16, 16, 16}, 2, MIOPEN_SOFTMAX_ACCURATE},     // small case (all stride 1)
        {{16, 16, 16}, 0, MIOPEN_SOFTMAX_LOG},          // small case (not last dim)
        {{16, 16, 16}, 2, MIOPEN_SOFTMAX_LOG},          // small case (all stride 1)
        {{1024, 1024, 32}, 0, MIOPEN_SOFTMAX_ACCURATE}, // large case (not last dim)
        {{1024, 1024, 32}, 2, MIOPEN_SOFTMAX_ACCURATE}, // large case (all stride 1)
        {{1024, 1024, 32}, 0, MIOPEN_SOFTMAX_LOG},      // large case (not last dim)
        {{1024, 1024, 32}, 2, MIOPEN_SOFTMAX_LOG}       // large case (all stride 1)
    };
}

inline std::vector<SoftmaxTestCase> GenFullTestCasesBackward()
{
    return {
        {{16, 16, 16}, 0, MIOPEN_SOFTMAX_ACCURATE},     // small case (not last dim)
        {{16, 16, 16}, 2, MIOPEN_SOFTMAX_ACCURATE},     // small case (all stride 1)
        {{16, 16, 16}, 0, MIOPEN_SOFTMAX_LOG},          // small case (not last dim)
        {{16, 16, 16}, 2, MIOPEN_SOFTMAX_LOG},          // small case (all stride 1)
        {{1024, 1024, 32}, 0, MIOPEN_SOFTMAX_ACCURATE}, // large case (not last dim)
        {{1024, 1024, 32}, 2, MIOPEN_SOFTMAX_ACCURATE}, // large case (all stride 1)
        {{1024, 1024, 32}, 0, MIOPEN_SOFTMAX_LOG},      // large case (not last dim)
        {{1024, 1024, 32}, 2, MIOPEN_SOFTMAX_LOG},      // large case (all stride 1)
        {{4, 32, 32}, 0, MIOPEN_SOFTMAX_ACCURATE},      // reduce small dim
        {{4, 32, 32}, 1, MIOPEN_SOFTMAX_LOG}            // reduce small dim
    };
}

template <typename T>
struct SoftmaxFwdTest : public ::testing::TestWithParam<SoftmaxTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        softmax_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dim       = softmax_config.dim;
        algorithm = softmax_config.algorithm;

        auto in_dims = softmax_config.input_dim;
        input        = tensor<T>{in_dims}.generate(gen_value);

        output = tensor<T>{in_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{in_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_softmax_contiguous_forward(input, ref_output, dim, algorithm);
        status = miopen::SoftmaxForward_V3(
            handle, input.desc, input_dev.get(), output.desc, output_dev.get(), dim, algorithm);

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
        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        auto error = miopen::rms_range(ref_output, output);

        EXPECT_LT(error, threshold * 10) << "Error output beyond tolerance Error: " << error
                                         << ",  Tolerance: " << threshold * 10;
    }

    SoftmaxTestCase softmax_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    uint32_t dim;
    miopenSoftmaxAlgorithm_t algorithm;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename T>
struct SoftmaxBwdTest : public ::testing::TestWithParam<SoftmaxTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        softmax_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dim       = softmax_config.dim;
        algorithm = softmax_config.algorithm;

        auto in_grad_dims = softmax_config.input_dim;
        input_grad        = tensor<T>{in_grad_dims};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        output_grad = tensor<T>{in_grad_dims}.generate(gen_value);
        output      = tensor<T>{in_grad_dims}.generate(gen_value);

        ref_input_grad = tensor<T>{in_grad_dims};
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), static_cast<T>(0));

        input_grad_dev  = handle.Write(input_grad.data);
        output_dev      = handle.Write(output.data);
        output_grad_dev = handle.Write(output_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_softmax_contiguous_backward(output, output_grad, ref_input_grad, dim, algorithm);
        status = miopen::SoftmaxBackward_V3(handle,
                                            output.desc,
                                            output_dev.get(),
                                            output_grad.desc,
                                            output_grad_dev.get(),
                                            input_grad.desc,
                                            input_grad_dev.get(),
                                            dim,
                                            algorithm);

        ASSERT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        auto error = miopen::rms_range(ref_input_grad, input_grad);

        EXPECT_LT(error, threshold * 10) << "Error output beyond tolerance Error: " << error
                                         << ",  Tolerance: " << threshold * 10;
    }

    SoftmaxTestCase softmax_config;

    tensor<T> input_grad;
    tensor<T> output;
    tensor<T> output_grad;

    tensor<T> ref_input_grad;

    uint32_t dim;
    miopenSoftmaxAlgorithm_t algorithm;

    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
};
