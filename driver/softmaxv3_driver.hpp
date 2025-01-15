/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include "mloSoftmaxHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include <../test/verify.hpp>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>

template <typename Tgpu, typename Tcheck>
int mloSoftmaxV3ForwardContiguousRunHost(miopenTensorDescriptor_t inputDesc,
                                         const Tgpu* input,
                                         Tcheck* outputHost,
                                         uint32_t dim,
                                         miopenSoftmaxAlgorithm_t algorithm)
{
    auto input_len      = miopen::deref(inputDesc).GetLengths();
    auto reduce_size    = input_len[dim];
    uint64_t inner_size = 1;
    for(size_t i = dim + 1; i < input_len.size(); i++)
    {
        inner_size *= input_len[i];
    }
    uint64_t outer_size = 1;
    for(uint32_t i = 0; i < dim; i++)
    {
        outer_size *= input_len[i];
    }

    for(uint64_t o = 0; o < outer_size; o++)
    {
        for(uint64_t i = 0; i < inner_size; i++)
        {
            Tgpu pmax       = std::numeric_limits<Tgpu>::min();
            size_t base_idx = o * reduce_size * inner_size + i;
            for(uint64_t r = 0; r < reduce_size; r++)
            {
                pmax = std::max(pmax, input[base_idx + r * inner_size]);
            }

            double psum = 0;
            for(uint64_t r = 0; r < reduce_size; r++)
            {
                double val = exp(static_cast<double>(input[base_idx + r * inner_size]) -
                                 static_cast<double>(pmax));
                psum += val;
            }

            for(uint64_t r = 0; r < reduce_size; r++)
            {
                outputHost[base_idx + r * inner_size] =
                    (algorithm == MIOPEN_SOFTMAX_LOG)
                        ? input[base_idx + r * inner_size] - pmax - log(psum)
                        : exp(input[base_idx + r * inner_size] - pmax) / psum;
            }
        }
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int mloSoftmaxV3BackwardContiguousRunHost(miopenTensorDescriptor_t outDesc,
                                          const Tgpu* output,
                                          miopenTensorDescriptor_t outGradDesc,
                                          const Tgpu* outGrad,
                                          miopenTensorDescriptor_t inGradDesc,
                                          Tcheck* inGrad,
                                          uint32_t dim,
                                          miopenSoftmaxAlgorithm_t algorithm)
{
    auto output_len     = miopen::deref(outDesc).GetLengths();
    auto reduce_size    = output_len[dim];
    uint64_t inner_size = 1;
    for(size_t i = dim + 1; i < output_len.size(); i++)
    {
        inner_size *= output_len[i];
    }
    uint64_t outer_size = 1;
    for(uint32_t i = 0; i < dim; i++)
    {
        outer_size *= output_len[i];
    }

    for(uint64_t o = 0; o < outer_size; o++)
    {
        for(uint64_t i = 0; i < inner_size; i++)
        {
            double psum = 0;
            for(uint64_t r = 0; r < reduce_size; r++)
            {
                if(algorithm == 2)
                {
                    psum += outGrad[i];
                }
                else
                {
                    psum += outGrad[i] * output[i];
                }
            }

            for(uint64_t r = 0; r < reduce_size; r++)
            {
                if(algorithm == 2)
                {
                    inGrad[i] = outGrad[i] - psum * exp(output[i]);
                }
                else
                {
                    inGrad[i] = (outGrad[i] - psum) * output[i];
                }
            }
        }
    }

    return 0;
}

template <typename Tgpu, typename Tref>
class SoftmaxV3Driver : public Driver
{
public:
    SoftmaxV3Driver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);
        miopenCreateTensorDescriptor(&inGradTensor);
        miopenCreateTensorDescriptor(&outGradTensor);

        data_type = miopen_type<Tgpu>{};
    }

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
    ~SoftmaxV3Driver() override
    {
        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(inGradTensor);
        miopenDestroyTensorDescriptor(outGradTensor);
    }

private:
    InputFlags inflags;

    int forw;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;
    // Backwards
    miopenTensorDescriptor_t inGradTensor;
    miopenTensorDescriptor_t outGradTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;
    // Backwards
    std::unique_ptr<GPUMem> inGrad_dev;
    std::unique_ptr<GPUMem> outGrad_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;
    // Backwards
    std::vector<Tgpu> inGrad;
    std::vector<Tgpu> outGrad;
    std::vector<Tref> inGradHost;

    miopenSoftmaxAlgorithm_t algorithm;
    uint32_t dim;
};

template <typename Tgpu, typename Tref>
int SoftmaxV3Driver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");
    if(forw != 0 && forw != 1)
    {
        MIOPEN_THROW("Invalid Forward|Backward Mode");
    }

    dim       = inflags.GetValueInt("dim");
    algorithm = static_cast<miopenSoftmaxAlgorithm_t>(inflags.GetValueInt("algorithm"));

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxV3Driver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = inflags.GetValueTensor("input_lengths").lengths;
    SetTensorNd(inputTensor, in_len, data_type);
    SetTensorNd(outputTensor, in_len, data_type);

    if(forw == 0)
    {
        SetTensorNd(inGradTensor, in_len, data_type);
        SetTensorNd(outGradTensor, in_len, data_type);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxV3Driver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "1",
                         "Run only Forward (1) or Run both Forward and Backward (0) (Default=1)",
                         "int");
    inflags.AddTensorFlag(
        "input_lengths", 'I', "2x10", "The dimensional lengths of the input tensor");
    inflags.AddInputFlag(
        "dim", 'D', "1", "The dimension which softmax is computed (Default=0)", "int");
    inflags.AddInputFlag(
        "algorithm",
        'A',
        "1",
        "Softmax Algorithm: Softmax Fast(0) | Softmax Accurate(1) | Softmax Log(2) (Default=1)",
        "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxV3Driver<Tgpu, Tref>::AllocateBuffersAndCopy()
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
        {
            std::cerr << "Error copying (input) to GPU, size: " << in_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }

        if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        {
            std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
    }

    if(forw == 0)
    {
        size_t out_sz     = GetTensorSpace(outputTensor);
        size_t inGrad_sz  = GetTensorSpace(inGradTensor);
        size_t outGrad_sz = GetTensorSpace(outGradTensor);

        // GPU allocation
        out_dev     = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
        inGrad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, inGrad_sz, sizeof(Tgpu)));
        outGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, outGrad_sz, sizeof(Tgpu)));

        // GPU host allocation
        out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
        inGrad  = std::vector<Tgpu>(inGrad_sz, static_cast<Tgpu>(0));
        outGrad = std::vector<Tgpu>(outGrad_sz, static_cast<Tgpu>(0));

        // CPU allocation
        inGradHost = std::vector<Tref>(inGrad_sz, static_cast<Tref>(0));

        for(int i = 0; i < outGrad_sz; i++)
        {
            outGrad[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        for(int i = 0; i < out_sz; i++)
        {
            out[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        {
            std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }

        if(outGrad_dev->ToGPU(GetStream(), outGrad.data()) != 0)
        {
            std::cerr << "Error copying (output gradient) to GPU, size: " << outGrad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }

        if(inGrad_dev->ToGPU(GetStream(), inGrad.data()) != 0)
        {
            std::cerr << "Error copying (input gradient) to GPU, size: " << inGrad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxV3Driver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenSoftmaxForward_V3(GetHandle(),
                                                        inputTensor,
                                                        in_dev->GetMem(),
                                                        outputTensor,
                                                        out_dev->GetMem(),
                                                        dim,
                                                        algorithm);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenSoftmaxForward_V3");

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
            std::cout << "Wall-clock Time Forward SoftmaxV3 Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward SoftmaxV3 Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
    {
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxV3Driver<Tgpu, Tref>::RunForwardCPU()
{
    mloSoftmaxV3ForwardContiguousRunHost(inputTensor, in.data(), outhost.data(), dim, algorithm);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxV3Driver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;
    Timer t;
    START_TIME;
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenSoftmaxBackward_V3(GetHandle(),
                                                         outputTensor,
                                                         out_dev->GetMem(),
                                                         outGradTensor,
                                                         outGrad_dev->GetMem(),
                                                         inGradTensor,
                                                         inGrad_dev->GetMem(),
                                                         dim,
                                                         algorithm);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenSoftmaxBackward_V3");

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
            std::cout << "Wall-clock Time Backward Softmax Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Softmax Elapsed: " << kernel_average_time << " ms\n";
    }

    if(inGrad_dev->FromGPU(GetStream(), inGrad.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << inGrad_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref SoftmaxV3Driver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int SoftmaxV3Driver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outhost, out);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Softmax FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Softmax Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxV3Driver<Tgpu, Tref>::RunBackwardCPU()
{
    mloSoftmaxV3BackwardContiguousRunHost(outputTensor,
                                          out.data(),
                                          outGradTensor,
                                          outGrad.data(),
                                          inGradTensor,
                                          inGradHost.data(),
                                          dim,
                                          algorithm);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxV3Driver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(inGradHost, inGrad);
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward Softmax FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward Softmax Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
