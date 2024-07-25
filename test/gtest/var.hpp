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
#include "cpu_var.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/var.hpp>

struct VarTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    int32_t* dims;
    int32_t num_dims;
    bool keepdim;
    bool unbiased;
    int32_t divisor;

    friend std::ostream& operator<<(std::ostream& os, const VarTestCase& tc)
    {
        return os << "N: " << tc.N << " C: " << tc.C << " D: " << tc.D << " H: " << tc.H
                  << " W: " << tc.W << " num_dims: " << tc.num_dims << " keepdim: " << tc.keepdim
                  << " unbiased: " << tc.unbiased << " divisor: " << tc.divisor;
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>{N, C, D, H, W};
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>{N, C, H, W};
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>{N, C, W};
        }
        else if((N != 0) && (C != 0))
        {
            return std::vector<size_t>{N, C};
        }
        else if((N != 0))
        {
            return std::vector<size_t>{N};
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }

    std::vector<int32_t> GetDims()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            std::vector<int32_t> adjusted_dims;
            for(int i = 0; i < num_dims; i++)
            {
                adjusted_dims.push_back(dims[i]);
            }
            return adjusted_dims;
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            std::vector<int32_t> adjusted_dims;
            for(int i = 0; i < num_dims; i++)
            {
                int32_t dim = dims[i];
                if(dim == 3 || dim == 4)
                {
                    adjusted_dims.push_back(dim - 1);
                }
                else if(dim == 2)
                {
                    std::cout << "Incorrect Dims\n" << std::endl;
                    return std::vector<int32_t>({0});
                }
                else
                {
                    adjusted_dims.push_back(dim);
                }
            }
            return adjusted_dims;
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            std::vector<int32_t> adjusted_dims;
            for(int i = 0; i < num_dims; i++)
            {
                int32_t dim = dims[i];
                if(dim == 4)
                {
                    adjusted_dims.push_back(dim - 2);
                }
                else if(dim == 3 || dim == 2)
                {
                    std::cout << "Incorrect Dims\n" << std::endl;
                    return std::vector<int32_t>({0});
                }
                else
                {
                    adjusted_dims.push_back(dim);
                }
            }
            return adjusted_dims;
        }
        else if((N != 0) && (C != 0))
        {
            std::vector<int32_t> adjusted_dims;
            for(int i = 0; i < num_dims; i++)
            {
                int32_t dim = dims[i];
                if(dim == 2 || dim == 3 || dim == 4)
                {
                    std::cout << "Incorrect Dims\n" << std::endl;
                    return std::vector<int32_t>({0});
                }
                else
                {
                    adjusted_dims.push_back(dim);
                }
            }
            return adjusted_dims;
        }
        else if((N != 0))
        {
            std::vector<int32_t> adjusted_dims;
            for(int i = 0; i < num_dims; i++)
            {
                int32_t dim = dims[i];
                if(dim == 1 || dim == 2 || dim == 3 || dim == 4)
                {
                    std::cout << "Incorrect Dims\n" << std::endl;
                    return std::vector<int32_t>({0});
                }
                else
                {
                    adjusted_dims.push_back(dim);
                }
            }
            return adjusted_dims;
        }
        else
        {
            std::cout << "Incorrect Dims\n" << std::endl;
            return std::vector<int32_t>({0});
        }
    }
};

std::vector<VarTestCase> VarTestConfigs()
{
    return {
        {2048, 0, 0, 0, 0, new int32_t[1]{0}, 1, true, true, 2048},
        {8192, 0, 0, 0, 0, new int32_t[1]{0}, 1, true, false, 8192},
        {65536, 0, 0, 0, 0, new int32_t[1]{0}, 1, true, false, 65536},
        {80, 40, 0, 0, 0, new int32_t[1]{0}, 1, true, true, 80},
        {80, 250, 0, 0, 0, new int32_t[1]{0}, 1, false, false, 80},
        {30, 300, 0, 0, 0, new int32_t[1]{1}, 1, true, false, 300},
        {30, 40, 0, 0, 0, new int32_t[2]{0, 1}, 2, false, true, 1200},
        {40, 50, 0, 0, 30, new int32_t[1]{0}, 1, false, true, 40},
        {30, 50, 0, 0, 10, new int32_t[1]{4}, 1, false, false, 10},
        {50, 40, 0, 0, 50, new int32_t[1]{4}, 1, true, true, 50},
        {40, 60, 0, 0, 10, new int32_t[2]{0, 4}, 2, true, true, 400},
        {20, 30, 0, 0, 40, new int32_t[2]{1, 4}, 2, true, true, 1200},
        {50, 30, 0, 0, 10, new int32_t[3]{0, 1, 4}, 3, true, true, 15000},
        {50, 10, 0, 5, 20, new int32_t[1]{0}, 1, false, false, 50},
        {50, 30, 0, 5, 4, new int32_t[1]{3}, 1, true, false, 5},
        {5, 100, 0, 10, 3, new int32_t[1]{4}, 1, false, false, 3},
        {10, 20, 0, 10, 5, new int32_t[2]{0, 4}, 2, false, true, 100},
        {20, 30, 0, 5, 5, new int32_t[2]{0, 3}, 2, true, true, 100},
        {20, 5, 0, 10, 10, new int32_t[3]{0, 1, 3}, 3, true, false, 1000},
        {30, 10, 0, 15, 10, new int32_t[3]{1, 3, 4}, 3, false, true, 1500},
        {10, 10, 0, 10, 10, new int32_t[4]{0, 1, 3, 4}, 4, false, false, 10000},
        {30, 10, 5, 10, 2, new int32_t[1]{0}, 1, false, false, 30},
        {30, 5, 20, 5, 4, new int32_t[1]{3}, 1, true, false, 5},
        {20, 10, 3, 12, 4, new int32_t[2]{0, 2}, 2, false, true, 60},
        {20, 5, 10, 5, 10, new int32_t[2]{0, 3}, 2, true, true, 100},
        {40, 20, 3, 2, 5, new int32_t[3]{0, 1, 2}, 3, false, false, 2400},
        {15, 3, 5, 20, 12, new int32_t[3]{1, 2, 3}, 3, true, false, 300},
        {12, 12, 4, 8, 10, new int32_t[4]{1, 2, 3, 4}, 4, true, true, 3840},
        {5, 5, 5, 10, 10, new int32_t[4]{0, 1, 2, 3}, 4, false, false, 1250},
        {10, 4, 8, 2, 4, new int32_t[5]{0, 1, 2, 3, 4}, 5, true, true, 2560},
    };
}

template <typename T = float>
struct VarBackwardTest : public ::testing::TestWithParam<VarTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        var_config     = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        std::vector<int32_t> dims_vector = var_config.GetDims();

        dims     = var_config.dims;
        num_dims = var_config.num_dims;

        for(int i = 0; i < var_config.num_dims; i++)
        {
            dims[i] = dims_vector[i];
        }

        keepdim  = var_config.keepdim;
        unbiased = var_config.unbiased;
        divisor  = var_config.divisor;

        // Calculate the tensor's dimensions
        auto input_dims = var_config.GetInput();

        std::vector<size_t> input_grad_dims(input_dims.size());
        std::copy(input_dims.begin(), input_dims.end(), input_grad_dims.begin());

        std::vector<size_t> mean_dims(input_dims.size());
        std::copy(input_dims.begin(), input_dims.end(), mean_dims.begin());

        std::vector<size_t> mean_grad_dims(input_dims.size());
        std::copy(input_dims.begin(), input_dims.end(), mean_grad_dims.begin());

        std::vector<size_t> var_grad_dims(input_dims.size());
        std::copy(input_dims.begin(), input_dims.end(), var_grad_dims.begin());

        for(const auto& dim : dims_vector)
        {
            mean_dims[dim]      = 1;
            mean_grad_dims[dim] = 1;
            var_grad_dims[dim]  = 1;
        }

        // Set up tensor's values
        input = tensor<T>{input_dims}.generate(gen_value);
        mean  = tensor<T>{mean_dims};
        cpu_mean(input, mean, dims_vector, divisor);
        mean_grad = tensor<T>{mean_grad_dims}.generate(gen_value);
        var_grad  = tensor<T>{var_grad_dims}.generate(gen_value);

        input_grad = tensor<T>{input_grad_dims};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input_grad = tensor<T>{input_grad_dims};
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

        cpu_var_backward(
            input, ref_input_grad, mean, mean_grad, var_grad, dims, num_dims, unbiased, divisor);
        miopenStatus_t status;

        status = miopen::VarBackward(handle,
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
                                     dims,
                                     num_dims,
                                     keepdim,
                                     unbiased,
                                     divisor);

        EXPECT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;
        auto error = miopen::rms_range(ref_input_grad, input_grad);

        EXPECT_TRUE(miopen::range_distance(ref_input_grad) == miopen::range_distance(input_grad));
        EXPECT_TRUE(error < threshold)
            << "Error output beyond tolerance Error:" << error << ",    Threshold " << threshold;
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

    int32_t* dims;
    int32_t num_dims;

    bool keepdim;
    bool unbiased;
    int32_t divisor;
};
