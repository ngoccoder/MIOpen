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

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>
#include <miopen/allocator.hpp>
#include <miopen/miopen.h>
#include <miopen/gather.hpp>

struct GatherNDTestCase
{
    std::vector<size_t> paramGradDims;
    std::vector<size_t> indicesDims;

    friend std::ostream& operator<<(std::ostream& os, const GatherNDTestCase& tc)
    {
        os << "Param dim: ";
        for(auto paramDim : tc.paramGradDims)
        {
            os << paramDim << " ";
        }
        os << std::endl;

        os << "Indices dim: ";
        for(auto indexDim : tc.indicesDims)
        {
            os << indexDim << " ";
        }
        os << std::endl;

        return os;
    }

    std::vector<size_t> GetParamGradDim() const { return paramGradDims; }
    std::vector<size_t> GetIndicesDim() const { return indicesDims; }

    GatherNDTestCase() {}

    GatherNDTestCase(std::vector<size_t> param_grad_dim, std::vector<size_t> indices_dim)
        : paramGradDims(param_grad_dim), indicesDims(indices_dim)
    {
    }
};

inline std::vector<GatherNDTestCase> GenFullTestCases()
{
    return {
        GatherNDTestCase({2, 2, 8}, {2, 6, 8, 1}),       // index_depth = 1
        GatherNDTestCase({16, 16, 3}, {24, 36, 1}),      // index_depth = 1
        GatherNDTestCase({10, 20, 3}, {12, 24, 2}),      // index_depth = 2
        GatherNDTestCase({16, 128, 256}, {16, 2}),       // index_depth = 2
        GatherNDTestCase({16, 256, 16, 72}, {16, 3}),    // index_depth = 3
        GatherNDTestCase({32, 40, 40, 64}, {2, 6, 8, 3}) // index_depth = 3
    };
}

template <typename T, typename I>
struct GatherNDBwdTest : public ::testing::TestWithParam<GatherNDTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto param_grad_dims = config.GetParamGradDim();
        auto indices_dims    = config.GetIndicesDim();
        auto indices_num_dim = indices_dims.size();
        size_t slice_dim     = (indices_num_dim > 1) ? indices_dims[indices_num_dim - 1] : 1;

        std::vector<size_t> out_grad_dims;
        for(size_t i = 0; i < indices_num_dim - 1; i++)
        {
            out_grad_dims.push_back(indices_dims[i]);
        }
        for(size_t i = slice_dim; i < param_grad_dims.size(); i++)
        {
            out_grad_dims.push_back(param_grad_dims[i]);
        }

        gatherDesc.setMode(MIOPEN_GATHER_ND);

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto min_dim_param_grad = *std::min_element(param_grad_dims.begin(), param_grad_dims.end());
        auto gen_index          = [min_dim_param_grad](auto...) {
            return prng::gen_0_to_B(static_cast<I>(min_dim_param_grad));
        };

        indices    = tensor<I>{indices_dims}.generate(gen_index);
        outputGrad = tensor<T>{out_grad_dims}.generate(gen_value);

        paramGrad = tensor<T>{param_grad_dims};
        std::fill(paramGrad.begin(), paramGrad.end(), static_cast<T>(0));

        ref_paramGrad = tensor<T>{param_grad_dims};
        std::fill(ref_paramGrad.begin(), ref_paramGrad.end(), static_cast<T>(0));

        paramGrad_dev  = handle.Write(paramGrad.data);
        indices_dev    = handle.Write(indices.data);
        outputGrad_dev = handle.Write(outputGrad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_gathernd_backward<T, I>(outputGrad, indices, ref_paramGrad);

        miopenStatus_t status;
        status = gatherDesc.Backward(handle,
                                     outputGrad.desc,
                                     outputGrad_dev.get(),
                                     indices.desc,
                                     indices_dev.get(),
                                     paramGrad.desc,
                                     paramGrad_dev.get());

        ASSERT_EQ(status, miopenStatusSuccess);

        paramGrad.data = handle.Read<T>(paramGrad_dev, paramGrad.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        double threshold = GetTolerance();
        ASSERT_EQ(miopen::range_distance(ref_paramGrad), miopen::range_distance(paramGrad));
        auto error = miopen::rms_range(ref_paramGrad, paramGrad);
        EXPECT_LT(error, threshold) << "Error output (param grad) beyond tolerance Error:" << error
                                    << ",  Threshold: " << threshold << std::endl;
    }

    GatherNDTestCase config;

    miopen::gather::GatherDescriptor gatherDesc;
    tensor<T> paramGrad;
    tensor<T> outputGrad;
    tensor<I> indices;

    tensor<T> ref_paramGrad;

    miopen::Allocator::ManageDataPtr paramGrad_dev;
    miopen::Allocator::ManageDataPtr outputGrad_dev;
    miopen::Allocator::ManageDataPtr indices_dev;
};
