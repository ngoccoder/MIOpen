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

#pragma once

#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/tensor_view_utils.hpp"
#include "tensor_driver.hpp"
#include "tensor_view.hpp"
#include "timer.hpp"
#include "random.hpp"
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloOuterForwardRunHost(const miopenTensorDescriptor_t x1Desc,
                               const miopenTensorDescriptor_t x2Desc,
                               const miopenTensorDescriptor_t yDesc,
                               const Tgpu* x1,
                               const Tgpu* x2,
                               Tcheck* y)
{
    auto y_tv    = miopen::get_inner_expanded_tv<2>(miopen::deref(yDesc));
    auto y_numel = miopen::deref(yDesc).GetElementSize();

    for(auto i = 0; i < y_numel; i++)
    {
        tensor_layout_t<2> y_layout(y_tv, i);
        y[y_tv.get_tensor_view_idx(y_layout)] = static_cast<Tcheck>(x1[y_layout.layout[0]]) *
                                                static_cast<Tcheck>(x2[y_layout.layout[1]]);
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloOuterBackwardRunHost(const miopenTensorDescriptor_t x1Desc,
                                const miopenTensorDescriptor_t x2Desc,
                                const miopenTensorDescriptor_t x1GradDesc,
                                const miopenTensorDescriptor_t x2GradDesc,
                                const miopenTensorDescriptor_t yGradDesc,
                                const Tgpu* x1,
                                const Tgpu* x2,
                                const Tgpu* yGrad,
                                Tcheck* x1Gradhost,
                                Tcheck* x2Gradhost)
{
    auto y_grad_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(yGradDesc));
    auto x1_numel  = miopen::deref(x1Desc).GetElementSize();
    auto x2_numel  = miopen::deref(x2Desc).GetElementSize();

    for(size_t i = 0; i < x1_numel; i++)
    {
        Tcheck sum = static_cast<Tcheck>(0.0f);
        for(size_t j = 0; j < x2_numel; j++)
        {
            sum += static_cast<Tcheck>(x2[j]) *
                   static_cast<Tcheck>(yGrad[y_grad_tv.get_tensor_view_idx({i, j})]);
        }
        x1Gradhost[i] = sum;
    }

    for(size_t j = 0; j < x2_numel; j++)
    {
        Tcheck sum = static_cast<Tcheck>(0.0f);
        for(size_t i = 0; i < x1_numel; i++)
        {
            sum += static_cast<Tcheck>(x1[i]) *
                   static_cast<Tcheck>(yGrad[y_grad_tv.get_tensor_view_idx({i, j})]);
        }
        x2Gradhost[j] = sum;
    }

    return 0;
}

template <typename Tgpu, typename Tref>
class OuterDriver : public Driver
{
public:
    OuterDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&x1Desc);
        miopenCreateTensorDescriptor(&x2Desc);
        miopenCreateTensorDescriptor(&yDesc);

        miopenCreateTensorDescriptor(&x1GradDesc);
        miopenCreateTensorDescriptor(&x2GradDesc);
        miopenCreateTensorDescriptor(&yGradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~OuterDriver() override
    {
        miopenDestroyTensorDescriptor(x1Desc);
        miopenDestroyTensorDescriptor(x2Desc);
        miopenDestroyTensorDescriptor(yDesc);

        miopenDestroyTensorDescriptor(x1GradDesc);
        miopenDestroyTensorDescriptor(x2GradDesc);
        miopenDestroyTensorDescriptor(yGradDesc);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

    miopenTensorDescriptor_t x1Desc;
    miopenTensorDescriptor_t x2Desc;
    miopenTensorDescriptor_t x1GradDesc;
    miopenTensorDescriptor_t x2GradDesc;
    miopenTensorDescriptor_t yDesc;
    miopenTensorDescriptor_t yGradDesc;

    std::unique_ptr<GPUMem> x1_dev;
    std::unique_ptr<GPUMem> x2_dev;
    std::unique_ptr<GPUMem> x1Grad_dev;
    std::unique_ptr<GPUMem> x2Grad_dev;
    std::unique_ptr<GPUMem> y_dev;
    std::unique_ptr<GPUMem> yGrad_dev;

    std::vector<Tgpu> x1;
    std::vector<Tgpu> x2;
    std::vector<Tgpu> x1Grad;
    std::vector<Tgpu> x2Grad;
    std::vector<Tgpu> y;
    std::vector<Tgpu> yGrad;

    std::vector<Tref> x1Gradhost;
    std::vector<Tref> x2Gradhost;
    std::vector<Tref> yhost;
};

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");

    if(forw != 0 && forw != 1 && forw != 2)
    {
        MIOPEN_THROW("Invalid Forward|Backward Mode");
    }

    isContiguous = inflags.GetValueInt("is_contiguous") == 0 ? false : true;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> x1_lens = inflags.GetValueTensor("x1_dim").lengths;
    SetTensorNd(x1Desc, x1_lens, data_type);

    std::vector<int> x2_lens = inflags.GetValueTensor("x2_dim").lengths;
    SetTensorNd(x2Desc, x2_lens, data_type);

    if(forw == 0 || forw == 1)
    {
        std::vector<int> y_lens({x1_lens[0], x2_lens[0]});
        auto y_stride = ComputeStrides(y_lens);
        SetTensorNd(yDesc, y_lens, y_stride, data_type);
    }

    if(forw == 0 || forw == 2)
    {
        SetTensorNd(x1GradDesc, x1_lens, data_type);
        SetTensorNd(x2GradDesc, x2_lens, data_type);

        std::vector<int> y_grad_lens({x1_lens[0], x2_lens[0]});
        auto y_grad_stride = ComputeStrides(y_grad_lens);
        SetTensorNd(yGradDesc, y_grad_lens, y_grad_stride, data_type);
    }

    return 0;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run both Forward and Backward (0), Run only Forward (1), Run only "
                         "Backward (2) (Default=0)",
                         "int");
    inflags.AddTensorFlag(
        "x1_dim", 'N', "32", "The dimensional lengths of first input tensor (Default=32)");
    inflags.AddInputFlag(
        "x2_dim", 'M', "32", "The dimensional lengths of second input tensor (Default=32)", "int");
    inflags.AddInputFlag(
        "is_contiguous", 'C', "1", "Is Tensor Contiguous (1) or not (0) (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> OuterDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    size_t x1_sz = GetTensorSize(x1Desc);
    size_t x2_sz = GetTensorSize(x2Desc);
    size_t y_sz  = GetTensorSize(yDesc);

    x1_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, x1_sz, sizeof(Tgpu)));
    x2_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, x2_sz, sizeof(Tgpu)));

    x1 = std::vector<Tgpu>(x1_sz, static_cast<Tgpu>(0));
    x2 = std::vector<Tgpu>(x2_sz, static_cast<Tgpu>(0));

    for(int i = 0; i < x1_sz; i++)
    {
        x1[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    for(int i = 0; i < x2_sz; i++)
    {
        x2[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    if(x1_dev->ToGPU(GetStream(), x1.data()) != 0)
        std::cerr << "Error copying (x1) to GPU, size: " << x1_dev->GetSize() << std::endl;

    if(x2_dev->ToGPU(GetStream(), x2.data()) != 0)
        std::cerr << "Error copying (in1) to GPU, size: " << x2_dev->GetSize() << std::endl;

    if(forw == 0 || forw == 1)
    {
        y_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, y_sz, sizeof(Tgpu)));

        y     = std::vector<Tgpu>(y_sz, static_cast<Tgpu>(0));
        yhost = std::vector<Tref>(y_sz, static_cast<Tref>(0));
    }

    if(forw == 0 || forw == 2)
    {
        x1Grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, x1_sz, sizeof(Tgpu)));
        x2Grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, x2_sz, sizeof(Tgpu)));
        yGrad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, y_sz, sizeof(Tgpu)));

        x1Grad = std::vector<Tgpu>(x1_sz, static_cast<Tgpu>(0));
        x2Grad = std::vector<Tgpu>(x2_sz, static_cast<Tgpu>(0));
        yGrad  = std::vector<Tgpu>(y_sz, static_cast<Tgpu>(0));

        x1Gradhost = std::vector<Tref>(x1_sz, static_cast<Tgpu>(0));
        x2Gradhost = std::vector<Tref>(x2_sz, static_cast<Tgpu>(0));

        for(int i = 0; i < y_sz; i++)
        {
            yGrad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(yGrad_dev->ToGPU(GetStream(), yGrad.data()) != 0)
            std::cerr << "Error copying (yGrad) to GPU, size: " << yGrad_dev->GetSize()
                      << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenOuterForward(GetHandle(),
                           x1Desc,
                           x1_dev->GetMem(),
                           x2Desc,
                           x2_dev->GetMem(),
                           yDesc,
                           y_dev->GetMem());

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward Outer Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Outer Elapsed: " << kernel_average_time << " ms\n";
    }

    if(y_dev->FromGPU(GetStream(), y.data()) != 0)
        std::cerr << "Error copying (y_dev) from GPU, size: " << y_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloOuterForwardRunHost<Tgpu, Tref>(x1Desc, x2Desc, yDesc, x1.data(), x2.data(), yhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenOuterBackward(GetHandle(),
                            x1Desc,
                            x1_dev->GetMem(),
                            x2Desc,
                            x2_dev->GetMem(),
                            x1GradDesc,
                            x1Grad_dev->GetMem(),
                            x2GradDesc,
                            x2Grad_dev->GetMem(),
                            yGradDesc,
                            yGrad_dev->GetMem());

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);

        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward Outer Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Outer Elapsed: " << kernel_average_time << " ms\n";
    }

    if(x1Grad_dev->FromGPU(GetStream(), x1Grad.data()) != 0)
        std::cerr << "Error copying (x1Grad_dev) from GPU, size: " << x1Grad_dev->GetSize()
                  << std::endl;

    if(x2Grad_dev->FromGPU(GetStream(), x2Grad.data()) != 0)
        std::cerr << "Error copying (x2Grad_dev) from GPU, size: " << x2Grad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloOuterBackwardRunHost<Tgpu, Tref>(x1Desc,
                                        x2Desc,
                                        x1GradDesc,
                                        x2GradDesc,
                                        yGradDesc,
                                        x1.data(),
                                        x2.data(),
                                        yGrad.data(),
                                        x1Gradhost.data(),
                                        x2Gradhost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref OuterDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(yhost, y);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Outer FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Outer Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int OuterDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error1          = miopen::rms_range(x1Gradhost, x1Grad);
    auto error2          = miopen::rms_range(x2Gradhost, x2Grad);

    if(!std::isfinite(error1) || error1 > tolerance)
    {
        std::cout << "Backward Outer FAILED with in1: " << error1 << " > " << tolerance
                  << std::endl;
        return EC_VerifyBwd;
    }
    else if(!std::isfinite(error2) || error2 > tolerance)
    {
        std::cout << "Backward Outer FAILED with in2: " << error2 << " > " << tolerance
                  << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward Outer Verifies OK on CPU reference (" << error1 << " < " << tolerance
                  << ')' << " and "
                  << "(" << error2 << " < " << tolerance << ')' << std::endl;
    }
    return miopenStatusSuccess;
}
