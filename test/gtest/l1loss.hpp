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

#include "cpu_l1loss.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/l1loss.hpp>

struct L1LossTestCase
{
    std::vector<size_t> dims;
    miopenL1LossReduction_t reduction;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const L1LossTestCase& tc)
    {
        os << "Dims: ";
        for(auto dim_sz : tc.dims)
        {
            os << dim_sz << " ";
        }
        return os << " reducion mode: " << tc.reduction << " contiguous: " << tc.isContiguous;
    }

    L1LossTestCase() {}

    L1LossTestCase(std::vector<size_t> dims_, miopenL1LossReduction_t reduction_, bool cont_)
        : dims(dims_), reduction(reduction_), isContiguous()(cont_)
    {
    }

    std::vector<size_t> ComputeStrides() const
    {
        std::vector<size_t> inputDim = dims;
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

inline std::vector<L1LossTestCase> L1LossTestConfigs()
{ // n c d h w dim
    // clang-format off
    return {
        {{1, 1, 1, 1, 1}, MIOPEN_L1LOSS_SUM_REDUCTION, false},
        {{1, 2, 3, 4, 1}, MIOPEN_L1LOSS_SUM_REDUCTION, false},
        {{1, 1, 1, 257, 1}, MIOPEN_L1LOSS_SUM_REDUCTION, false},
        {{2, 10, 128, 128, 1}, MIOPEN_L1LOSS_SUM_REDUCTION, false},
        {{5, 13, 17, 11, 1}, MIOPEN_L1LOSS_MEAN_REDUCTION, false},
        {{256, 4, 8723, 1, 1}, MIOPEN_L1LOSS_SUM_REDUCTION, false},
        {{256, 4, 8723, 1, 1}, MIOPEN_L1LOSS_SUM_REDUCTION, true},
        {{1, 1, 1, 1, 1}, MIOPEN_L1LOSS_SUM_REDUCTION, true},
        {{34, 4, 5, 1, 1}, MIOPEN_L1LOSS_SUM_REDUCTION, true},
        {{4, 7, 5, 1, 1}, MIOPEN_L1LOSS_SUM_REDUCTION, true},
        {{15, 4, 5, 1, 1}, MIOPEN_L1LOSS_SUM_REDUCTION, true}
    };
    // clang-format on
}

template <typename T = float>
struct L1LossFwdTest : public ::testing::TestWithParam<L1LossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        l1loss_config   = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 1); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 2); };

        reduction       = l1loss_config.reduction;
        auto in_dims    = l1loss_config.dims;
        auto in_strides = l1loss_config.ComputeStrides();
        input           = tensor<T>{in_dims, in_strides}.generate(gen_value1);

        auto tar_strides = l1loss_config.ComputeStrides();
        target           = tensor<T>{in_dims, tar_strides}.generate(gen_value2);

        auto out_lengths =
            (reduction == MIOPEN_L1LOSS_NONE_REDUCTION) ? in_dims : std::vector<size_t>{1};

        output = tensor<T>{out_lengths};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_lengths};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_lengths;
        ws_sizeInBytes = (reduction == MIOPEN_L1LOSS_NONE_REDUCTION)
                             ? 0
                             : miopen::GetL1LossForwardWorkspaceSize(
                                   handle, reduction, input.desc, target.desc, output.desc);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(T));

            workspace = tensor<T>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), static_cast<T>(0));

            ref_workspace = tensor<T>{workspace_dims};
            std::fill(ref_workspace.begin(), ref_workspace.end(), static_cast<T>(0));

            workspace_dev = handle.Write(workspace.data);
        }

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        if(reduction != MIOPEN_L1LOSS_NONE_REDUCTION)
        {
            cpu_l1loss_reduced_forward<T>(input, target, ref_output, ref_workspace, reduction);
            status         = miopen::L1LossForward(handle,
                                           reduction,
                                           workspace_dev.get(),
                                           ws_sizeInBytes,
                                           input.desc,
                                           input_dev.get(),
                                           target.desc,
                                           target_dev.get(),
                                           output.desc,
                                           output_dev.get());
            workspace.data = handle.Read<T>(workspace_dev, workspace.data.size());
        }

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

        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error: " << error
                                            << ",  Tolerance: " << threshold * 10;
    }

    L1LossTestCase l1loss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<T> workspace;
    miopenL1LossReduction_t reduction;

    tensor<T> ref_workspace;
    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};
