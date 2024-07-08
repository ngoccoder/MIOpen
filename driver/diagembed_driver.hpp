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
#ifndef GUARD_MIOPEN_DIAGEMBED_DRIVER_HPP
#define GUARD_MIOPEN_DIAGEMBED_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "miopen/diagonal/solvers.hpp"
#include "miopen/diagonal/diagembed/problem_description.hpp"
#include "miopen/errors.hpp"
#include "miopen/tensor_view_utils.hpp"
#include "tensor_driver.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include "random.hpp"
#include "tensor_view.hpp"
#include "timer.hpp"
#include "../test/verify.hpp"

#ifndef MLO_DIAGEMBEDHOST_H_
#define MLO_DIAGEMBEDHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloDiagEmbedForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                   Tgpu* input,
                                   miopenTensorDescriptor_t outputDesc,
                                   Tcheck* outputHost,
                                   int64_t offset,
                                   int64_t dim1,
                                   int64_t dim2)
{
    auto diag_tv =
        miopen::solver::diagonal::getDiagonal(miopen::deref(outputDesc), offset, dim1, dim2);
    auto input_tv    = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto input_numel = miopen::deref(inputDesc).GetElementSize();

    for(size_t i = 0; i < input_numel; i++)
    {
        auto layout   = tensor_layout_t<5>(input_tv, i);
        auto inputIdx = input_tv.get_tensor_view_idx(layout);

        auto outLayout = tensor_layout_t<5>(diag_tv, i);
        auto outIdx    = diag_tv.get_tensor_view_idx(outLayout);

        outputHost[outIdx] = input[inputIdx];
    }

    return 0;
}

#endif

template <typename Tgpu, typename Tref>
class DiagEmbedDriver : public Driver
{
public:
    DiagEmbedDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> inputDim);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int SetBNParametersFromCmdLineArgs();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU(); // Verify implements it

    int RunBackwardGPU() override;
    int RunBackwardCPU(); // Verify implements it

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~DiagEmbedDriver() override
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    int64_t offset;
    int64_t dim1;
    int64_t dim2;
};

template <typename Tgpu, typename Tref>
int DiagEmbedDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagEmbedDriver<Tgpu, Tref>::GetandSetData()
{
    SetBNParametersFromCmdLineArgs();

    std::vector<int> in_len = GetInputTensorLengthsFromCmdLine();
    offset                  = inflags.GetValueInt("Offset");
    dim1                    = inflags.GetValueInt("Dim1");
    dim2                    = inflags.GetValueInt("Dim2");
    isContiguous            = inflags.GetValueInt("contiguous") > 0 ? true : false;
    auto numDim             = in_len.size();

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

    auto inStride = ComputeStrides(in_len);
    SetTensorNd(inputTensor, in_len, inStride, data_type);
    auto out_len = in_len;

    auto new_dim_len = abs(offset) + in_len[numDim - 1];
    out_len.pop_back();
    auto insert_pos = std::min(dim1, dim2);
    if(insert_pos > out_len.size())
    {
        insert_pos = out_len.size();
    }
    out_len.insert(out_len.begin() + insert_pos, new_dim_len);
    insert_pos = std::max(dim1, dim2);
    if(insert_pos > out_len.size())
    {
        insert_pos = out_len.size();
    }
    out_len.insert(out_len.begin() + insert_pos, new_dim_len);

    SetTensorNd(outputTensor, out_len, data_type);

    return miopenStatusSuccess;
}

// Equivalent tensor.transpose(0, -1).contiguous().transpose(0, -1)
template <typename Tgpu, typename Tref>
std::vector<int> DiagEmbedDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int DiagEmbedDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (1) (Default = 1)", "int");
    inflags.AddInputFlag("batchsize", 'n', "2", "Mini-batch size (Default=2)", "int");
    inflags.AddInputFlag("in_channels", 'c', "0", "Number of Input Channels (Default=0)", "int");
    inflags.AddInputFlag("in_d", 'D', "0", "Input Depth (Default=0)", "int");
    inflags.AddInputFlag("in_h", 'H', "0", "Input Height (Default=0)", "int");
    inflags.AddInputFlag("in_w", 'W', "4", "Input Width (Default=4)", "int");

    inflags.AddInputFlag(
        "Offset", 'R', "0", "Control which diagonal to consider (Default=0)", "int");
    inflags.AddInputFlag("Dim1",
                         'A',
                         "-2",
                         "First dimension with respect to which to take diagonal (Default=-2)",
                         "int");
    inflags.AddInputFlag("Dim2",
                         'B',
                         "-1",
                         "Second dimension with respect to which to take diagonal (Default=-1)",
                         "int");
    inflags.AddInputFlag("contiguous", 'C', "1", "Tensor is contiguous or not", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> DiagEmbedDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    int in_n = inflags.GetValueInt("batchsize");
    int in_c = inflags.GetValueInt("in_channels");
    int in_d = inflags.GetValueInt("in_d");
    int in_h = inflags.GetValueInt("in_h");
    int in_w = inflags.GetValueInt("in_w");

    if((in_n != 0) && (in_c != 0) && (in_d != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_h != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
    else if((in_n != 0) && (in_c != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_c, in_w});
    }
    else if((in_n != 0) && (in_w != 0))
    {
        return std::vector<int>({in_n, in_w});
    }
    else if(in_n != 0)
    {
        return std::vector<int>({in_n});
    }
    else
    {
        std::cerr << "Error Input Tensor Lengths\n" << std::endl;
        return std::vector<int>({0});
    }
}

template <typename Tgpu, typename Tref>
int DiagEmbedDriver<Tgpu, Tref>::SetBNParametersFromCmdLineArgs()
{
    forw = inflags.GetValueInt("forw");
    if(forw != 1)
    {
        printf("Incorrect Forward Mode\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagEmbedDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    if(forw == 1)
    {
        size_t in_sz  = GetTensorSpace(inputTensor);
        size_t out_sz = GetTensorSpace(outputTensor);

        // GPU allocation
        in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
        out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

        // GPU host allocation
        in  = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
        out = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

        // CPU allocation
        outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

        for(int i = 0; i < in_sz; i++)
        {
            in[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(in_dev->ToGPU(GetStream(), in.data()) != 0)
            std::cerr << "Error copying (input) to GPU, size: " << in_dev->GetSize() << std::endl;

        if(out_dev->ToGPU(GetStream(), out.data()) != 0)
            std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagEmbedDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenDiagEmbedForward(GetHandle(),
                               inputTensor,
                               in_dev->GetMem(),
                               outputTensor,
                               out_dev->GetMem(),
                               offset,
                               dim1,
                               dim2);

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
            std::cout << "Wall-clock Time Forward Diag Embed Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Diag Embed Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagEmbedDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloDiagEmbedForwardRunHost<Tgpu, Tref>(
        inputTensor, in.data(), outputTensor, outhost.data(), offset, dim1, dim2);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagEmbedDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref DiagEmbedDriver<Tgpu, Tref>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int DiagEmbedDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Diag Embed FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Diag Embed Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagEmbedDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int DiagEmbedDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_DIAGEMBED_DRIVER_HPP
