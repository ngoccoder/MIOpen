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

#include "cpu_var.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/var.hpp>

struct VarTestCase
{
    std::vector<size_t> input_dim;
    std::vector<int> dims;
    bool unbiased;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const VarTestCase& tc)
    {
        os << "Input dims: ";
        for(auto dim_sz : tc.input_dim)
        {
            os << dim_sz << " ";
        }
        os << "Dims: ";
        for(auto dim : tc.dims)
        {
            os << dim << " ";
        }
        return os << " unbiased: " << tc.unbiased << " contiguous: " << tc.isContiguous;
    }

    VarTestCase() {}

    VarTestCase(std::vector<size_t> input_dim_,
                std::vector<int> dims_,
                bool unbiased_,
                bool isCont_)
        : input_dim(input_dim_), dims(dims_), unbiased(unbiased_), isContiguous(isCont_)
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

std::vector<VarTestCase> GenFullTestCases()
{
    return {{{2048}, {0}, true, false},
            {{8192}, {0}, true, true},
            {{65536}, {0}, true, true},
            {{80, 40}, {0, 1}, true, true},
            {{80, 250}, {0, 1}, false, true},
            {{30, 300}, {1}, true, true},
            {{30, 40}, {0, 1}, false, true},
            {{40, 50, 30}, {0}, false, true},
            {{30, 50, 10}, {0, 1}, false, true},
            {{50, 40, 50}, {0}, true, true},
            {{50, 40, 50}, {0}, true, false},
            {{40, 60, 10}, {0, 2}, true, true},
            {{40, 60, 10}, {0, 2}, true, false},
            {{30, 10, 5, 10, 2}, {0, 1, 2}, false, true},
            {{30, 10, 5, 10, 2}, {0, 1, 2}, true, false}};
}

template <typename T = float>
struct VarBwdTest : public ::testing::TestWithParam<VarTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        var_config      = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 99); };
        auto gen_value3 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 98); };
        auto gen_value4 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 97); };

        dims     = var_config.dims;
        unbiased = var_config.unbiased;

        // Calculate the tensor's dimensions
        auto input_dims = var_config.input_dim;

        std::vector<size_t> input_grad_dims = input_dims;
        std::vector<size_t> mean_dims       = input_dims;
        std::vector<size_t> mean_grad_dims  = input_dims;
        std::vector<size_t> var_grad_dims   = input_dims;

        for(const auto& dim : dims)
        {
            mean_dims[dim]      = 1;
            mean_grad_dims[dim] = 1;
            var_grad_dims[dim]  = 1;
        }

        // Set up tensor's values
        auto input_stride = var_config.ComputeStrides(input_dims);
        input             = tensor<T>{input_dims, input_stride}.generate(gen_value1);
        mean              = tensor<T>{mean_dims};
        cpu_mean(input, mean, dims);
        mean_grad = tensor<T>{mean_grad_dims}.generate(gen_value3);
        var_grad  = tensor<T>{var_grad_dims}.generate(gen_value4);

        auto input_grad_stride = var_config.ComputeStrides(input_grad_dims);
        input_grad             = tensor<T>{input_grad_dims, input_grad_stride};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input_grad = tensor<T>{input_grad_dims, input_grad_stride};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev      = handle.Write(input.data);
        input_grad_dev = handle.Write(input_grad.data);
        mean_dev       = handle.Write(mean.data);
        mean_grad_dev  = handle.Write(mean_grad.data);
        var_grad_dev   = handle.Write(var_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        dim_5d_t dims_onehot;
        for(auto dim : dims)
        {
            dims_onehot.x[dim] = 1;
        }

        cpu_var_backward(input, ref_input_grad, mean, mean_grad, var_grad, dims_onehot, unbiased);
        miopenStatus_t status;

        status = miopen::var::VarBackward(handle,
                                          input.desc,
                                          input_dev.get(),
                                          input_grad.desc,
                                          input_grad_dev.get(),
                                          mean.desc,
                                          mean_dev.get(),
                                          mean_grad.desc,
                                          mean_grad_dev.get(),
                                          var_grad.desc,
                                          var_grad_dev.get(),
                                          dims.data(),
                                          dims.size(),
                                          unbiased);

        EXPECT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        auto threshold = std::numeric_limits<T>::epsilon() * 10;
        auto error     = miopen::rms_range(ref_input_grad, input_grad);

        EXPECT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error, threshold);
    }

    VarTestCase var_config;

    tensor<T> input;
    tensor<T> input_grad;
    tensor<T> mean;
    tensor<T> mean_grad;
    tensor<T> var_grad;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr mean_dev;
    miopen::Allocator::ManageDataPtr mean_grad_dev;
    miopen::Allocator::ManageDataPtr var_grad_dev;

    std::vector<int> dims;
    bool unbiased;
};
