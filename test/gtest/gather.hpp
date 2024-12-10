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

#include "cpu_gather.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <miopen/allocator.hpp>
#include <miopen/gather.hpp>
#include <miopen/miopen.h>

struct GatherTestCase
{
    std::vector<size_t> inputDims;
    std::vector<size_t> indicesDims;
    uint32_t dim;

    friend std::ostream& operator<<(std::ostream& os, const GatherTestCase& tc)
    {
        os << "Input dim: ";
        for(auto inputDim : tc.inputDims)
        {
            os << inputDim << " ";
        }
        os << std::endl;

        os << "Indices dim: ";
        for(auto indexDim : tc.indicesDims)
        {
            os << indexDim << " ";
        }
        os << std::endl;

        os << "dim: " << tc.dim;

        return os;
    }

    std::vector<size_t> GetInputDim() const { return inputDims; }
    std::vector<size_t> GetIndicesDim() const { return indicesDims; }
    uint32_t GetDim() const { return dim; }

    GatherTestCase() {}

    GatherTestCase(std::vector<size_t> input_dim, std::vector<size_t> indices_dim, uint32_t dim_)
        : inputDims(input_dim), indicesDims(indices_dim), dim(dim_)
    {
    }
};

inline std::vector<GatherTestCase> GenFullTestCases()
{
    return {
        GatherTestCase({5, 6, 7}, {5, 6, 7}, 0),         // same shape
        GatherTestCase({15, 30, 45}, {15, 30, 45}, 2),   // same shape
        GatherTestCase({2, 2, 3}, {2, 2, 2}, 1),         // different shape
        GatherTestCase({16, 16, 3}, {24, 2, 2}, 0),      // different shape
        GatherTestCase({16, 256, 256}, {16, 256, 16}, 1) // large size
    };
}

template <typename T>
struct GatherFwdTest : public ::testing::TestWithParam<GatherTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto input_dims   = config.GetInputDim();
        auto indices_dims = config.GetIndicesDim();
        dim               = config.GetDim();

        std::vector<size_t> output_dims = indices_dims;

        gatherDesc.setMode(MIOPEN_GATHER);
        gatherDesc.setDim(dim);

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto dim_size  = input_dims[dim];
        auto gen_index = [dim_size](auto...) {
            return prng::gen_0_to_B(static_cast<int64_t>(dim_size));
        };

        indices = tensor<uint64_t>{indices_dims}.generate(gen_index);
        input   = tensor<T>{input_dims}.generate(gen_value);

        output = tensor<T>{output_dims};
        std::fill(output.begin(), output.end(), static_cast<T>(0));

        ref_output = tensor<T>{output_dims};
        std::fill(ref_output.begin(), ref_output.end(), static_cast<T>(0));

        input_dev   = handle.Write(input.data);
        indices_dev = handle.Write(indices.data);
        output_dev  = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_gather_forward<T>(input, indices, ref_output, dim);

        miopenStatus_t status;
        status = gatherDesc.Forward(handle,
                                    input.desc,
                                    input_dev.get(),
                                    indices.desc,
                                    indices_dev.get(),
                                    output.desc,
                                    output_dev.get());

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
        auto error       = miopen::rms_range(ref_output, output);
        EXPECT_LT(error, threshold) << "Error output (param grad) beyond tolerance Error:" << error
                                    << ",  Threshold: " << threshold << std::endl;
    }

    GatherTestCase config;

    miopen::GatherDescriptor gatherDesc;
    tensor<T> input;
    tensor<T> output;
    tensor<uint64_t> indices;

    tensor<T> ref_output;

    uint32_t dim;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
};
