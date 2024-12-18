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

#include "cpu_trace.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include "random.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <limits>

#include <miopen/miopen.h>
#include <miopen/trace.hpp>

struct TraceTestCase
{
    std::vector<size_t> input_dim;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const TraceTestCase& tc)
    {
        os << "Input dims: ";
        for(auto dim_sz : tc.input_dim)
        {
            os << dim_sz << " ";
        }
        return os << " contiguous: " << tc.isContiguous;
    }

    TraceTestCase() {}

    TraceTestCase(std::vector<size_t> input_dims_, bool cont_)
        : input_dim(input_dims_), isContiguous(cont_)
    {
    }

    std::vector<size_t> ComputeStrides() const
    {
        std::vector<size_t> inputDim = input_dim;
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

inline std::vector<TraceTestCase> GenFullTestCases()
{ // n c d h w dim
    // clang-format off
    return {
        {{1, 8}, false},                      // non-cont small case
        {{8, 4}, false},                      // non-cont small case
        {{512, 512}, false},                  // non-cont large case 
        {{384, 640}, false},                  // non-cont large case
        {{512, 768}, true},                   // cont large case
        {{1024, 1024}, true},                 // cont large case
        {{1, 10}, true},                      // cont small case
        {{34, 20}, true},                     // cont small case
    };
    // clang-format on
}

template <typename T>
struct TraceFwdTest : public ::testing::TestWithParam<TraceTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        trace_config    = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 99); };

        auto in_dims    = trace_config.input_dim;
        auto in_strides = trace_config.ComputeStrides();
        input           = tensor<T>{in_dims, in_strides}.generate(gen_value1);

        auto out_lengths = std::vector<size_t>{1};

        output = tensor<T>{out_lengths};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_lengths};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        ws_sizeInBytes =
            miopen::trace::GetTraceForwardWorkspaceSize(handle, input.desc, output.desc);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(float));

            workspace = tensor<float>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), static_cast<float>(0));

            workspace_dev = handle.Write(workspace.data);
        }

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_trace_forward<T>(input, ref_output);
        status = miopen::trace::TraceForward(handle,
                                             workspace_dev.get(),
                                             ws_sizeInBytes,
                                             input.desc,
                                             input_dev.get(),
                                             output.desc,
                                             output_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

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
        auto error       = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Tolerance: " << threshold * 10;
    }

    TraceTestCase trace_config;

    tensor<T> input;
    tensor<T> output;
    tensor<float> workspace;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};

template <typename T>
struct TraceBwdTest : public ::testing::TestWithParam<TraceTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        trace_config    = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 99); };

        auto in_grad_dims    = trace_config.input_dim;
        auto in_grad_strides = trace_config.ComputeStrides();
        input_grad           = tensor<T>{in_grad_dims, in_grad_strides};

        auto out_grad_dims = std::vector<size_t>{1};
        output_grad        = tensor<T>{out_grad_dims}.generate(gen_value1);

        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input_grad = tensor<T>{in_grad_dims, in_grad_strides};
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), static_cast<T>(0));

        input_grad_dev  = handle.Write(input_grad.data);
        output_grad_dev = handle.Write(output_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_trace_backward<T>(output_grad, ref_input_grad);
        status = miopen::trace::TraceBackward(
            handle, output_grad.desc, output_grad_dev.get(), input_grad.desc, input_grad_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

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
        auto error       = miopen::rms_range(ref_input_grad, input_grad);

        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Tolerance: " << threshold * 10;
    }

    TraceTestCase trace_config;

    tensor<T> input_grad;
    tensor<T> output_grad;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
};
