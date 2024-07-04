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
#include "cpu_diagembed.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/diagonal.hpp>
#include <numeric>

struct DiagEmbedTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int64_t offset;
    int64_t dim1;
    int64_t dim2;
    friend std::ostream& operator<<(std::ostream& os, const DiagEmbedTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " offset:" << tc.offset << " dim1:" << tc.dim1
                  << " dim2:" << tc.dim2;
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

std::vector<DiagEmbedTestCase> DiagEmbedTestConfigs()
{ // n c d h w dim
    // clang-format off
    return {
        { 2,    0,   0,  0,   4,    0, 0, 1},
        { 2,    0,   0,  0,   4,    8, 0, 1},
        { 2,    0,   0,  0,   4,    -8, 2, 1},
        { 16,    0,   0,  0,   16,    0, 0, 1},
        { 16,    0,   0,  0,   16,    2, 0, 2},
        { 16,    0,   0,  0,   16,    -2, 0, 1},
        { 32,    0,   0,  0,    0,     0, 0, 1},
        { 32,    4,   0,  2,    4,     2, 0, 3},
      };
    // clang-format on
}

template <typename T>
struct DiagEmbedFwdTest : public ::testing::TestWithParam<DiagEmbedTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle    = get_handle();
        diagembed_config = GetParam();
        auto gen_value   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        offset = diagembed_config.offset;
        dim1   = diagembed_config.dim1;
        dim2   = diagembed_config.dim2;

        auto in_dims = diagembed_config.GetInput();
        auto numDim  = in_dims.size();

        input = tensor<T>{in_dims}.generate(gen_value);
        if(dim1 == dim2)
        {
            throw std::runtime_error("Dim1 and Dim2 cannot be the same");
        }

        if(dim1 < 0)
        {
            dim1 = numDim + dim1;
        }
        if(dim2 < 0)
        {
            dim2 = numDim + dim2;
        }
        if(dim1 < 0 || dim2 < 0)
        {
            throw std::runtime_error("Dimension out of range");
        }

        std::vector<size_t> out_dim = in_dims;
        auto new_dim_len            = abs(offset) + in_dims[numDim - 1];
        out_dim.pop_back();
        auto insert_pos = std::min(dim1, dim2);
        if(insert_pos > out_dim.size())
        {
            insert_pos = out_dim.size();
        }
        out_dim.insert(out_dim.begin() + insert_pos, new_dim_len);
        insert_pos = std::max(dim1, dim2);
        if(insert_pos > out_dim.size())
        {
            insert_pos = out_dim.size();
        }
        out_dim.insert(out_dim.begin() + insert_pos, new_dim_len);

        output = tensor<T>{out_dim};
        std::fill(output.begin(), output.end(), static_cast<T>(0));

        ref_output = tensor<T>{out_dim};
        std::fill(ref_output.begin(), ref_output.end(), static_cast<T>(0));

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_diagembed_forward(input, ref_output, offset, dim1, dim2);
        miopenStatus_t status;

        status = miopen::DiagEmbedForward(
            handle, input.desc, input_dev.get(), output.desc, output_dev.get(), offset, dim1, dim2);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
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
        double threshold = GetTolerance();
        auto error       = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }

    DiagEmbedTestCase diagembed_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    int64_t offset;
    int64_t dim1;
    int64_t dim2;
};
