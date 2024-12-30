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

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include <../test/verify.hpp>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>

template <typename Tgpu, typename Tcheck>
int mloEmbeddingBackward(const miopenTensorDescriptor_t inputDesc,
                         const int64_t* input,
                         const miopenTensorDescriptor_t outputGradDesc,
                         const Tgpu* outputGrad,
                         const miopenTensorDescriptor_t weightGradDesc,
                         Tcheck* weightGradHost,
                         const int32_t* indices_freq,
                         int64_t padding_idx)
{
    auto input_tv       = miopen::get_inner_expanded_tv<4>(miopen::deref(inputDesc));
    auto outGrad_tv     = miopen::get_inner_expanded_tv<4>(miopen::deref(outputGradDesc));
    auto weightGrad_tv  = miopen::get_inner_expanded_tv<2>(miopen::deref(weightGradDesc));
    auto weightGrad_len = miopen::deref(weightGradDesc).GetLengths();
    auto embedding_dim  = weightGrad_len[1];
    auto num_embeddings = weightGrad_len[0];
    auto outGrad_numel  = miopen::deref(outputGradDesc).GetElementSize();
    for(size_t o = 0; o < outGrad_numel; o++)
    {
        size_t i = o / embedding_dim, j = o % embedding_dim;
        size_t n3 = i % input_tv.size[3], n012 = i / input_tv.size[3];
        size_t n2 = n012 % input_tv.size[2], n01 = n012 / input_tv.size[2];
        size_t n1 = n01 % input_tv.size[1], n0 = n01 / input_tv.size[1];

        size_t input_idx      = input_tv.get_tensor_view_idx({n0, n1, n2, n3});
        int64_t embedding_idx = input[input_idx];

        if(embedding_idx == padding_idx)
            continue;

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            Tcheck scale =
                indices_freq
                    ? (static_cast<Tcheck>(1.0f) / static_cast<Tcheck>(indices_freq[input_idx]))
                    : 1.0f;
            size_t weight_grad_idx = weightGrad_tv.get_tensor_view_idx({embedding_idx, j});
            tensor_layout_t<4> outGrad_layout(outGrad_tv, o);
            // std::cout << "[cput] idx for weight grad: " << weight_grad_idx << " gid = " << o <<
            // "emb_idx = " << embedding_idx << " j = " << j << std::endl;
            weightGradHost[weight_grad_idx] +=
                outputGrad[outGrad_tv.get_tensor_view_idx(outGrad_layout)] * scale;
        }
    }

    return 0;
}

template <typename Tgpu, typename Tref>
class EmbeddingDriver : public Driver
{
public:
    EmbeddingDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensorGrad);
        miopenCreateTensorDescriptor(&weightTensorGrad);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU(); // Verify implements it

    int RunBackwardGPU() override;
    int RunBackwardCPU(); // Verify implements it

    Tref GetTolerance();

    int VerifyBackward() override;
    int VerifyForward() override;
    ~EmbeddingDriver() override
    {
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(outputTensorGrad);
        miopenDestroyTensorDescriptor(weightTensorGrad);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

    // Backwards
    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensorGrad;
    miopenTensorDescriptor_t weightTensorGrad;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> outGrad_dev;
    std::unique_ptr<GPUMem> weightGrad_dev;

    std::vector<int64_t> in;
    std::vector<Tgpu> outGrad;
    std::vector<Tgpu> weightGrad;
    std::vector<Tref> weightGradHost;

    std::vector<int32_t> indices_freq;
    std::unique_ptr<GPUMem> indices_freq_dev;

    int64_t padding_idx;
    bool scale_grad_by_freq;
};

template <typename Tgpu, typename Tref>
int EmbeddingDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");

    if(forw != 2)
    {
        MIOPEN_THROW("Invalid Forward|Backward Mode");
    }

    isContiguous       = inflags.GetValueInt("is_contiguous") == 1 ? true : false;
    padding_idx        = inflags.GetValueInt("padding_idx");
    scale_grad_by_freq = inflags.GetValueInt("scale_grad_by_freq") == 1 ? true : false;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int EmbeddingDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = inflags.GetValueTensor("input_dims").lengths;
    auto in_stride          = ComputeStrides(in_len);
    SetTensorNd(inputTensor, in_len, in_stride, data_type);

    auto weight_len = inflags.GetValueTensor("weight_dims").lengths;
    SetTensorNd(weightTensorGrad, weight_len, data_type);

    std::vector<int> out_len = in_len;
    out_len.push_back(weight_len.back());
    SetTensorNd(outputTensorGrad, out_len, data_type);

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> EmbeddingDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int EmbeddingDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "2", "Run only Backward (2) (Default=2)", "int");
    inflags.AddTensorFlag(
        "input_dims", 'I', "4x4", "The dimensional lengths of the input tensor (Default=256x512)");
    inflags.AddTensorFlag("weight_dims",
                          'W',
                          "33x1",
                          "The dimensional lengths of the weight tensor (Default=1024x1024)");
    inflags.AddInputFlag(
        "scale_grad_by_freq",
        'N',
        "0",
        "If true (1), scale gradients by the inverse of frequency of the words (Default=0)",
        "int");
    inflags.AddInputFlag("padding_idx", 'P', "0", "Padding Index (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag(
        "is_contiguous", 'C', "1", "Is Tensor Contiguous (1) or not (0) (Default=1)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int EmbeddingDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    size_t in_sz    = GetTensorSpace(inputTensor);
    auto weight_len = miopen::deref(weightTensorGrad).GetLengths();

    // TODO: add indices_freq
    if(forw == 2)
    {
        size_t outGrad_sz    = GetTensorSpace(outputTensorGrad);
        size_t weightGrad_sz = GetTensorSpace(weightTensorGrad);

        // GPU allocation
        in_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(int64_t)));
        outGrad_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, outGrad_sz, sizeof(Tgpu)));
        weightGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, weightGrad_sz, sizeof(Tgpu)));

        // GPU host allocation
        in         = std::vector<int64_t>(in_sz, static_cast<int64_t>(0));
        outGrad    = std::vector<Tgpu>(outGrad_sz, static_cast<Tgpu>(0));
        weightGrad = std::vector<Tgpu>(weightGrad_sz, static_cast<Tgpu>(0));

        // CPU allocation
        weightGradHost = std::vector<Tref>(weightGrad_sz, static_cast<Tref>(0));

        for(int i = 0; i < in_sz; i++)
        {
            in[i] = prng::gen_A_to_B(static_cast<int64_t>(0.0),
                                     static_cast<int64_t>(weight_len[0] - 1));
        }
        for(int i = 0; i < outGrad_sz; i++)
        {
            outGrad[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(in_dev->ToGPU(GetStream(), in.data()) != 0)
        {
            std::cerr << "Error copying (input) to GPU, size: " << in_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
        if(outGrad_dev->ToGPU(GetStream(), outGrad.data()) != 0)
        {
            std::cerr << "Error copying (output gradient) to GPU, size: " << outGrad_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }

        if(scale_grad_by_freq)
        {
            auto input_numel = miopen::deref(inputTensor).GetElementSize();
            indices_freq     = std::vector<int32_t>(in_sz, static_cast<int32_t>(0));

            std::unordered_map<int64_t, int> counts;
            for(auto idx : in)
            {
                counts[idx]++;
            }
            for(size_t i = 0; i < input_numel; i++)
            {
                indices_freq[i] = counts[in[i]];
            }

            indices_freq_dev =
                std::unique_ptr<GPUMem>(new GPUMem(ctx, input_numel, sizeof(int32_t)));
            if(indices_freq_dev->ToGPU(GetStream(), indices_freq.data()) != 0)
            {
                std::cerr << "Error copying (indices_freq) to GPU, size: "
                          << indices_freq_dev->GetSize() << std::endl;
                return miopenStatusInternalError;
            }
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int EmbeddingDriver<Tgpu, Tref>::RunForwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int EmbeddingDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int EmbeddingDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;
    Timer t;
    START_TIME;
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenEmbeddingBackward(GetHandle(),
                                                        inputTensor,
                                                        in_dev->GetMem(),
                                                        outputTensorGrad,
                                                        outGrad_dev->GetMem(),
                                                        weightTensorGrad,
                                                        weightGrad_dev->GetMem(),
                                                        scale_grad_by_freq,
                                                        indices_freq.data(),
                                                        padding_idx);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenEmbeddingBackward");

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
            std::cout << "Wall-clock Time Backward Embedding Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward Embedding Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(weightGrad_dev->FromGPU(GetStream(), weightGrad.data()) != 0)
    {
        std::cerr << "Error copying (weightGrad_dev) from GPU, size: " << weightGrad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref EmbeddingDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int EmbeddingDriver<Tgpu, Tref>::VerifyForward()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int EmbeddingDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloEmbeddingBackward(inputTensor,
                         in.data(),
                         outputTensorGrad,
                         outGrad.data(),
                         weightTensorGrad,
                         weightGradHost.data(),
                         indices_freq.data(),
                         padding_idx);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int EmbeddingDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(weightGradHost, weightGrad);

    // auto weightGradSize = weightGrad.size();
    // for(int i = 0; i < weightGradSize; i++)
    //{
    //    std::cout << "weightGradHost[" << i << "] = " << weightGradHost[i] << " weightGrad[" << i
    //              << "] = " << weightGrad[i] << std::endl;
    //}

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward Embedding FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward Embedding OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
