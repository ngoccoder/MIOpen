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
#include "tensor_driver.hpp"
#include "tensor_view.hpp"
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
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tcheck>
int mloCosineSimilarityForward(const miopenTensorDescriptor_t input1Desc,
                               const Tgpu* input1,
                               const miopenTensorDescriptor_t input2Desc,
                               const Tgpu* input2,
                               const miopenTensorDescriptor_t outputDesc,
                               Tcheck* output,
                               uint32_t dim,
                               float eps)
{
    auto out_sz                   = miopen::deref(outputDesc).GetElementSize();
    tensor_view_t<5> input1_tv    = miopen::get_inner_expanded_tv<5>(miopen::deref(input1Desc));
    tensor_view_t<5> input2_tv    = miopen::get_inner_expanded_tv<5>(miopen::deref(input2Desc));
    tensor_view_t<4> output_tv_4d = miopen::get_inner_expanded_tv<4>(miopen::deref(outputDesc));
    tensor_view_t<5> output_tv;

    uint64_t[4] out_size = output_tv_4d.size;
    std::vector<uint64_t> out_strides;

    for(auto o = 0; o < out_sz; o++)
    {
        Tgpu xy = 0;
        Tgpu xn = 0;
        Tgpu yn = 0;

        tensor_layout_t<5> out_layout(output_tv, o);

        // TODO: tensor view for dim into out_tv
        for(size_t k = 0; k < input1_tv.size[dim]; ++k)
        {
            Tgpu x = input1[input1_tv.get_tensor_view_idx(out_layout)];
            Tgpu y = input2[input2_tv.get_tensor_view_idx(out_layout)];

            xy += x * y;
            xn += x * x;
            yn += y * y;

            out_layout.layout[dim]++;
        }

        xn = xn > eps ? xn : eps;
        yn = yn > eps ? yn : eps;

        output[output_tv.get_tensor_view_idx(out_layout)] = xy / sqrt(xn * yn);
    }
}

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
    auto outGrad_tv     = miopen::get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    auto weightGrad_tv  = miopen::get_inner_expanded_tv<2>(miopen::deref(weightGradDesc));
    auto weightGrad_len = miopen::deref(weightGradDesc).GetLengths();
    auto embedding_dim  = weightGrad_len[1];
    auto num_embeddings = weightGrad_len[0];
    auto outGrad_numel  = miopen::deref(outputGradDesc).GetElementSize();
    for(size_t o = 0; o < outGrad_numel; o++)
    {
        size_t i = o / embedding_dim, j = o % embedding_dim;

        tensor_layout_t<4> input_layout(input_tv, i);
        size_t input_idx      = input_tv.get_tensor_view_idx(input_layout);
        int64_t embedding_idx = input[input_idx];

        if(embedding_idx == padding_idx)
            continue;

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            Tcheck scale =
                indices_freq
                    ? (static_cast<Tcheck>(1.0f) / static_cast<Tcheck>(indices_freq[input_idx]))
                    : static_cast<Tcheck>(1.0f);
            size_t weight_grad_idx = weightGrad_tv.get_tensor_view_idx({embedding_idx, j});
            tensor_layout_t<5> outGrad_layout(outGrad_tv, o);
            weightGradHost[weight_grad_idx] +=
                outputGrad[outGrad_tv.get_tensor_view_idx(outGrad_layout)] * scale;
        }
    }

    return 0;
}

template <typename Tgpu, typename Tref>
class CosineSimilarityDriver : public Driver
{
public:
    CosineSimilarityDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&input1Tensor);
        miopenCreateTensorDescriptor(&input2Tensor);
        miopenCreateTensorDescriptor(&outputTensor);

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
    ~CosineSimilarityDriver() override
    {
        miopenDestroyTensorDescriptor(input1Tensor);
        miopenDestroyTensorDescriptor(input2Tensor);
        miopenDestroyTensorDescriptor(outputTensor);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

    // Forwards
    miopenTensorDescriptor_t input1Tensor;
    miopenTensorDescriptor_t input2Tensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> in1_dev;
    std::unique_ptr<GPUMem> in2_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in1;
    std::vector<Tgpu> in2;
    std::vector<Tref> out;

    uint32_t dim;
    float eps;
};

template <typename Tgpu, typename Tref>
int CosineSimilarityDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");

    if(forw != 1)
    {
        MIOPEN_THROW("Invalid Forward|Backward Mode");
    }

    isContiguous = inflags.GetValueInt("is_contiguous") == 0 ? false : true;
    dim          = inflags.GetValueInt("dim");
    eps          = inflags.GetValueDouble("eps");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CosineSimilarityDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in1_len = inflags.GetValueTensor("input1_dims").lengths;
    auto in1_stride          = ComputeStrides(in1_len);
    SetTensorNd(input1Tensor, in1_len, in1_stride, data_type);

    std::vector<int> in2_len = inflags.GetValueTensor("input2_dims").lengths;
    auto in2_stride          = ComputeStrides(in2_len);
    SetTensorNd(input2Tensor, in2_len, in2_stride, data_type);

    std::vector<int> out_len;

    for(int i = 0; i < in1_len.size(); i++)
    {
        if(i != dim)
        {
            out_len.push_back(max(in1_len[i], in2_len[i]));
        }
    }
    auto out_stride = ComputeStrides(out_len);
    SetTensorNd(outputTensor, out_len, out_stride, data_type);

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> CosineSimilarityDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int CosineSimilarityDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (1) (Default=1)", "int");
    inflags.AddTensorFlag(
        "input1_dims", 'I', "40x40", "The dimensional lengths of the input tensor (Default=40x40)");
    inflags.AddTensorFlag("input2_dims",
                          'J',
                          "40x40",
                          "The dimensional lengths of the input2 tensor (Default=40x40)");
    inflags.AddInputFlag(
        "dim", 'D', "1", "Dimension where cosine similarity is computed (Default=1)");
    inflags.AddInputFlag(
        "eps", 'E', "1e-8", "Small value to avoid division by zero (Default=1e-8)");
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
int CosineSimilarityDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    size_t in1_sz = GetTensorSpace(input1Tensor);
    size_t in2_sz = GetTensorSpace(input2Tensor);
    size_t out_sz = GetTensorSpace(outputTensor);

    if(forw == 1)
    {
        // GPU allocation
        in1_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in1_sz, sizeof(Tgpu)));
        in2_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, in2_sz, sizeof(Tgpu)));
        out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

        // GPU host allocation
        in1 = std::vector<Tgpu>(in1_sz, static_cast<Tgpu>(0));
        in2 = std::vector<Tgpu>(in2_sz, static_cast<Tgpu>(0));

        // CPU allocation
        out = std::vector<Tgpu>(out_sz, static_cast<Tref>(0));

        for(int i = 0; i < in1_sz; i++)
        {
            in1[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        for(int i = 0; i < in2_sz; i++)
        {
            in2[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(in1_dev->ToGPU(GetStream(), in1.data()) != 0)
        {
            std::cerr << "Error copying (input1) to GPU, size: " << in1_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
        if(in2_dev->ToGPU(GetStream(), in2.data()) != 0)
        {
            std::cerr << "Error copying (input2) to GPU, size: " << in2_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
        if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        {
            std::cerr << "Error copying (output) to GPU, size: " << out_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CosineSimilarityDriver<Tgpu, Tref>::RunForwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int CosineSimilarityDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int CosineSimilarityDriver<Tgpu, Tref>::RunBackwardGPU()
{
    // float kernel_total_time = 0;
    // float kernel_first_time = 0;
    // Timer t;
    // START_TIME;
    // for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    //{
    //    miopenStatus_t status = miopenEmbeddingBackward(GetHandle(),
    //                                                    inputTensor,
    //                                                    in_dev->GetMem(),
    //                                                    outputTensorGrad,
    //                                                    outGrad_dev->GetMem(),
    //                                                    weightTensorGrad,
    //                                                    weightGrad_dev->GetMem(),
    //                                                    indices_freq.data(),
    //                                                    padding_idx);
    //
    //    MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenEmbeddingBackward");
    //
    //    float time = 0.0;
    //    miopenGetKernelTime(GetHandle(), &time);
    //    kernel_total_time += time;
    //    if(i == 0)
    //        kernel_first_time = time;
    //}
    //
    // if(inflags.GetValueInt("time") == 1)
    //{
    //    STOP_TIME
    //    int iter = inflags.GetValueInt("iter");
    //    if(WALL_CLOCK)
    //        std::cout << "Wall-clock Time Backward Embedding Elapsed: " << t.gettime_ms() / iter
    //                  << " ms\n";
    //    float kernel_average_time =
    //        iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
    //    std::cout << "GPU Kernel Time Backward Embedding Elapsed: " << kernel_average_time
    //              << " ms\n";
    //}
    //
    // if(weightGrad_dev->FromGPU(GetStream(), weightGrad.data()) != 0)
    //{
    //    std::cerr << "Error copying (weightGrad_dev) from GPU, size: " <<
    //    weightGrad_dev->GetSize()
    //              << std::endl;
    //    return miopenStatusInternalError;
    //}

    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
Tref CosineSimilarityDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int CosineSimilarityDriver<Tgpu, Tref>::VerifyForward()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int CosineSimilarityDriver<Tgpu, Tref>::RunBackwardCPU()
{
    // mloEmbeddingBackward(inputTensor,
    //                     in.data(),
    //                     outputTensorGrad,
    //                     outGrad.data(),
    //                     weightTensorGrad,
    //                     weightGradHost.data(),
    //                     indices_freq.data(),
    //                     padding_idx);

    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int CosineSimilarityDriver<Tgpu, Tref>::VerifyBackward()
{
    // RunBackwardCPU();
    // const Tref tolerance = GetTolerance();
    // auto error           = miopen::rms_range(weightGradHost, weightGrad);
    //
    // if(!std::isfinite(error) || error > tolerance)
    //{
    //    std::cout << "Backward Embedding FAILED: " << error << " > " << tolerance << std::endl;
    //    return EC_VerifyBwd;
    //}
    // else
    //{
    //    std::cout << "Backward Embedding OK on CPU reference (" << error << " < " << tolerance
    //              << ')' << std::endl;
    //}

    return miopenStatusNotImplemented;
}
