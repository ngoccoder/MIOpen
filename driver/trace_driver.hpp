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
#include "tensor_driver.hpp"
#include "tensor_view.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <cstddef>
#include <memory>
#include <vector>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tcheck>
int mloTraceForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                           const Tgpu* input,
                           Tcheck* outputHost)
{
    tensor_view_t<2> input_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(inputDesc));
    auto input_len            = miopen::deref(inputDesc).GetLengths();
    size_t N                  = std::min(input_len[0], input_len[1]);
    double res                = 0;
    for(size_t i = 0; i < N; i++)
    {
        tensor_layout_t<2> input_layout = {i, i};
        size_t input_idx                = input_tv.get_tensor_view_idx(input_layout);
        Tgpu val                        = input[input_idx];
        res += val;
    }
    outputHost[0] = static_cast<Tcheck>(res);

    return 0;
}

template <typename Tgpu, typename Tcheck>
int mloTraceBackwardRunHost(const miopenTensorDescriptor_t outputGradDesc,
                            const Tgpu* outputGrad,
                            const miopenTensorDescriptor_t inputGradDesc,
                            Tcheck* inputGradHost)
{
    tensor_view_t<2> input_grad_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(inputGradDesc));
    tensor_view_t<1> output_grad_tv =
        miopen::get_inner_expanded_tv<1>(miopen::deref(outputGradDesc));
    auto input_grad_len = miopen::deref(inputGradDesc).GetLengths();
    size_t N            = input_grad_len[0];

    par_ford(N)([&](size_t i) {
        size_t idx = i % (input_grad_tv.size[1] + 1);

        if(idx != input_grad_tv.size[1])
        {
            tensor_layout_t<1> outgrad_layout = {0};
            Tgpu val = outputGrad[output_grad_tv.get_tensor_view_idx(outgrad_layout)];
            tensor_layout_t<2> ingrad_layout = {i, idx};
            inputGradHost[input_grad_tv.get_tensor_view_idx(ingrad_layout)] =
                static_cast<Tcheck>(val);
        }
    });

    return 0;
}

template <typename Tgpu, typename Tref>
class TraceDriver : public Driver
{
public:
    TraceDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> inputDim);
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
    ~TraceDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t outputGradDesc;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    std::unique_ptr<GPUMem> in_grad_dev;
    std::unique_ptr<GPUMem> out_grad_dev;
    std::unique_ptr<GPUMem> workspace_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tgpu> in_grad;
    std::vector<Tgpu> out_grad;
    std::vector<Tgpu> workspace;

    std::vector<Tref> outHost;
    std::vector<Tref> inGradHost;

    size_t ws_sizeInBytes;
};

// Equivalent tensor.transpose(0, -1).contiguous().transpose(0, -1)
template <typename Tgpu, typename Tref>
std::vector<int> TraceDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int TraceDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    isContiguous = inflags.GetValueInt("contiguous") > 0 ? true : false;

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");

    if(forw != 0 && forw != 1)
    {
        MIOPEN_THROW("Invalid Forward Mode");
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TraceDriver<Tgpu, Tref>::GetandSetData()
{
    auto in_len     = inflags.GetValueTensor("dim_lengths").lengths;
    auto in_strides = ComputeStrides(in_len);
    SetTensorNd(inputDesc, in_len, in_strides, data_type);
    SetTensorNd(inputGradDesc, in_len, in_strides, data_type);

    std::vector<int> out_lens = {1};
    SetTensorNd(outputDesc, out_lens, data_type);
    SetTensorNd(outputGradDesc, out_lens, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TraceDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag(
        "forw",
        'F',
        "0",
        "Run only Trace Forward (1) or Run both Forward and Backward (0) (Default = 0)",
        "int");
    inflags.AddTensorFlag(
        "dim_lengths", 'D', "9x9", "The dimensional lengths of the input tensor (Default=9x9)");
    inflags.AddInputFlag("contiguous",
                         'C',
                         "1",
                         "Tensor is contiguous (1) or not contiguous (0) (Default=1)",
                         "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TraceDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t in_sz  = GetTensorSize(inputDesc);
    size_t out_sz = GetTensorSize(outputDesc);

    miopenGetTraceForwardWorkspaceSize(GetHandle(), inputDesc, outputDesc, &ws_sizeInBytes);

    if(ws_sizeInBytes == static_cast<size_t>(-1))
        return miopenStatusAllocFailed;

    size_t ws_sz = ws_sizeInBytes / sizeof(Tgpu);

    uint32_t ctx = 0;

    in_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    workspace_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, ws_sizeInBytes, sizeof(std::byte)));

    in_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in        = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out       = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    workspace = std::vector<Tgpu>(ws_sz, static_cast<Tgpu>(0));

    in_grad  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out_grad = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

    outHost    = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    inGradHost = std::vector<Tref>(in_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(0.2));
    }

    for(int i = 0; i < out_sz; i++)
    {
        out_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(0.3));
    }

    if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;

    if(out_grad_dev->ToGPU(GetStream(), out_grad.data()) != 0)
        std::cerr << "Error copying (out_grad) to GPU, size: " << out_grad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TraceDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenTraceForward(GetHandle(),
                                                   workspace_dev->GetMem(),
                                                   ws_sizeInBytes,
                                                   inputDesc,
                                                   in_dev->GetMem(),
                                                   outputDesc,
                                                   out_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenTraceForward");

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
            std::cout << "Wall-clock Time Forward Trace Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Trace Elapsed: " << kernel_average_time << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TraceDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloTraceForwardRunHost<Tgpu, Tref>(inputDesc, in.data(), outHost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TraceDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenTraceBackward(GetHandle(),
                                                    outputGradDesc,
                                                    out_grad_dev->GetMem(),
                                                    inputGradDesc,
                                                    in_grad_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenTraceBackward");

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
            std::cout << "Wall-clock Time Backward Trace Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Trace Elapsed: " << kernel_average_time << " ms\n";
    }

    if(in_grad_dev->FromGPU(GetStream(), in_grad.data()) != 0)
        std::cerr << "Error copying (in_grad_dev) from GPU, size: " << in_grad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TraceDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloTraceBackwardRunHost(outputGradDesc, out_grad.data(), inputGradDesc, inGradHost.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref TraceDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int TraceDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outHost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Trace FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Trace Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int TraceDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(inGradHost, in_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward Trace FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward Trace Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
