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

#include "../driver/tensor_driver.hpp"
#include "cpu_diag.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/diagonal.hpp>
#include <numeric>

struct DiagTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int64_t diagonal;
    friend std::ostream& operator<<(std::ostream& os, const DiagTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " diagonal:" << tc.diagonal;
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, D, H, W});
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, H, W});
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, W});
        }
        else if((N != 0) && (W != 0))
        {
            return std::vector<size_t>({N, W});
        }
        else if((N != 0))
        {
            return std::vector<size_t>({N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }
};

std::vector<DiagTestCase> DiagTestConfigs()
{ // n c d h w dim
    // clang-format off
    return {
        { 2,    0,   0,  0,   4,    0},
        { 2,    0,   0,  0,   4,    8},
        { 2,    0,   0,  0,   4,    -8},
        { 16,    0,   0,  0,   16,    0},
        { 16,    0,   0,  0,   16,    2},
        { 16,    0,   0,  0,   16,    -2},
        { 32,    0,   0,  0,    0,     0},
        { 32,    0,   0,  0,    0,     2},
        { 32,    0,   0,  0,    0,     -2},
        { 64,    0,  0,  0,    7,     0},
        { 64,    0,  0,  0,    7,     2},
        { 64,    0,  0,  0,    7,     -2},
      };
    // clang-format on
}

template <typename T>
struct DiagFwdTest : public ::testing::TestWithParam<DiagTestCase>
{
protected:
    void SetUp() override
    {

        auto&& handle  = get_handle();
        diag_config    = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        diagonal = diag_config.diagonal;

        auto in_dims = diag_config.GetInput();

        input = tensor<T>{in_dims}.generate(gen_value);

        std::vector<size_t> out_dims;

        if(input.desc.GetLengths().size() == 1)
        {
            size_t sz = in_dims[0] + abs(diagonal);
            out_dims  = {sz, sz};
        }
        else
        {
            int64_t sz = 0;
            if(diagonal >= 0)
            {
                sz = std::min(static_cast<int64_t>(in_dims[0]),
                              static_cast<int64_t>(in_dims[1]) - diagonal);
            }
            else
            {
                sz = std::min(static_cast<int64_t>(in_dims[0]) + diagonal,
                              static_cast<int64_t>(in_dims[1]));
            }

            if(sz <= 0)
            {
                isOutputRequired = false;
            }
            else
            {
                out_dims = {sz};
            }
        }

        if(isOutputRequired)
        {
            output = tensor<T>{out_dims};
            std::fill(output.begin(), output.end(), static_cast<T>(0));

            ref_output = tensor<T>{out_dims};
            std::fill(ref_output.begin(), ref_output.end(), static_cast<T>(0));
        }

        input_dev  = handle.Write(input.data);
        output_dev = isOutputRequired ? handle.Write(output.data) : nullptr;
    }

    void RunTest()
    {
        if(isOutputRequired)
        {
            auto&& handle = get_handle();
            cpu_diag_forward(input, ref_output, diagonal);
            miopenStatus_t status;

            status = miopen::DiagForward(
                handle, input.desc, input_dev.get(), output.desc, output_dev.get(), diagonal);

            EXPECT_EQ(status, miopenStatusSuccess);

            output.data = handle.Read<T>(output_dev, output.data.size());
        }
    }

    double GetTolerance()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;
        return tolerance;
    }

    void Verify()
    {
        if(isOutputRequired)
        {
            double threshold = GetTolerance();
            auto error       = miopen::rms_range(ref_output, output);

            EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
            EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                                << ",  Thresholdx10: " << threshold * 10;
        }
    }

    DiagTestCase diag_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    int64_t diagonal;
    bool isOutputRequired = true;
};
