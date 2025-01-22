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
int mloEmbeddingBagForward(const miopenTensorDescriptor_t inputDesc,
                           const int64_t* input,
                           const miopenTensorDescriptor_t weightDesc,
                           const Tgpu* weight,
                           const miopenTensorDescriptor_t offsetsDesc,
                           const int64_t* offsets,
                           const miopenTensorDescriptor_t perSampleWeightsDesc,
                           const Tgpu* per_sample_weights,
                           const miopenTensorDescriptor_t outputDesc,
                           Tcheck* output,
                           miopenEmbeddingBagMode_t mode)
{
    auto input_len             = miopen::deref(inputDesc).GetLengths();
    tensor_view_t<2> weight_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(weightDesc));
    int64_t num_embeddings     = weight_tv.size[0];
    tensor_view_t<2> output_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(outputDesc));
    auto output_numel          = miopen::deref(outputDesc).GetElementSize();

    if(input_len.size() == 1) // with offsets
    {
        tensor_view_t<1> input_tv   = miopen::get_inner_expanded_tv<1>(miopen::deref(inputDesc));
        tensor_view_t<1> offsets_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(offsetsDesc));
        for(auto o = 0; o < output_numel; o++)
        {
            tensor_layout_t<2> output_layout(output_tv, o);
            size_t bag         = output_layout.layout[0];
            size_t feature_dim = output_layout.layout[1];

            size_t input_start = offsets[bag];
            size_t input_end = (bag + 1 < offsets_tv.size[0]) ? offsets[bag + 1] : input_tv.size[0];
            auto divisor     = input_end - input_start;
            double res =
                (mode == MIOPEN_EMBEDDING_BAG_MAX) ? std::numeric_limits<double>::min() : 0;

            for(auto i = input_start; i < input_end; i++)
            {
                int64_t embedding_idx = input[i];

                if(embedding_idx >= 0 && embedding_idx < num_embeddings)
                {
                    double w = static_cast<double>(
                        weight[weight_tv.get_tensor_view_idx({embedding_idx, feature_dim})]);
                    if(mode == MIOPEN_EMBEDDING_BAG_MAX)
                    {
                        res = std::max(res, w);
                    }
                    else
                    {
                        Tgpu scale =
                            per_sample_weights ? per_sample_weights[i] : static_cast<Tgpu>(1);
                        res += w * static_cast<double>(scale);
                    }
                }
            }

            output[output_tv.get_tensor_view_idx(output_layout)] =
                (mode == MIOPEN_EMBEDDING_BAG_MEAN) ? (divisor ? (res / divisor) : 0) : res;
        }
    }
    else if(input_len.size() == 2) // no offsets
    {
        tensor_view_t<2> input_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(inputDesc));
        tensor_view_t<2> per_sample_weights_tv =
            miopen::get_inner_expanded_tv<2>(miopen::deref(perSampleWeightsDesc));
        for(auto o = 0; o < output_numel; o++)
        {
            tensor_layout_t<2> output_layout(output_tv, o);
            double res =
                (mode == MIOPEN_EMBEDDING_BAG_MAX) ? std::numeric_limits<double>::min() : 0;
            for(size_t i = 0; i < input_tv.size[1]; i++)
            {
                int64_t embedding_idx =
                    input[input_tv.get_tensor_view_idx({output_layout.layout[0], i})];
                if(embedding_idx >= 0 && embedding_idx < num_embeddings)
                {
                    double w = static_cast<double>(weight[weight_tv.get_tensor_view_idx(
                        {embedding_idx, output_layout.layout[1]})]);
                    if(mode == MIOPEN_EMBEDDING_BAG_MAX)
                    {
                        res = std::max(res, w);
                    }
                    else
                    {
                        Tgpu scale =
                            per_sample_weights
                                ? per_sample_weights[per_sample_weights_tv.get_tensor_view_idx(
                                      {output_layout.layout[0], i})]
                                : static_cast<Tgpu>(1);
                        res += w * static_cast<double>(scale);
                    }
                }
            }
            output[output_tv.get_tensor_view_idx(output_layout)] =
                (mode == MIOPEN_EMBEDDING_BAG_MEAN) ? res / input_tv.size[1] : res;
        }
    }

    return 0;
}

template <typename Tgpu, typename Tref>
class EmbeddingBagDriver : public Driver
{
public:
    EmbeddingBagDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&weightTensor);
        miopenCreateTensorDescriptor(&perSampleWeightsTensor);
        miopenCreateTensorDescriptor(&offsetsTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        data_type    = miopen_type<Tgpu>{};
        int64_t_type = miopen_type<int64_t>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();

    int VerifyBackward() override;
    int VerifyForward() override;
    ~EmbeddingBagDriver() override
    {
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(weightTensor);
        miopenDestroyTensorDescriptor(perSampleWeightsTensor);
        miopenDestroyTensorDescriptor(offsetsTensor);
        miopenDestroyTensorDescriptor(outputTensor);
    }

private:
    InputFlags inflags;

    miopenDataType_t int64_t_type;
    int forw;
    bool isContiguous;

    // Forward
    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t weightTensor;
    miopenTensorDescriptor_t perSampleWeightsTensor;
    miopenTensorDescriptor_t offsetsTensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> weight_dev;
    std::unique_ptr<GPUMem> per_sample_weights_dev;
    std::unique_ptr<GPUMem> offsets_dev;
    std::unique_ptr<GPUMem> output_dev;

    std::vector<int64_t> input;
    std::vector<Tgpu> weight;
    std::vector<Tgpu> per_sample_weights;
    std::vector<int64_t> offsets;
    std::vector<Tgpu> output;
    std::vector<Tref> outputHost;

    miopenEmbeddingBagMode_t mode;
};

template <typename Tgpu, typename Tref>
int EmbeddingBagDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
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
    mode         = static_cast<miopenEmbeddingBagMode_t>(inflags.GetValueInt("mode"));

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int EmbeddingBagDriver<Tgpu, Tref>::GetandSetData()
{
    auto in_len    = inflags.GetValueTensor("input_dims").lengths;
    auto in_stride = ComputeStrides(in_len);
    SetTensorNd(inputTensor, in_len, in_stride, int64_t_type);

    auto per_sample_weights_len = inflags.GetValueTensor("per_sample_weights_dims").lengths;
    SetTensorNd(perSampleWeightsTensor, per_sample_weights_len, data_type);

    auto offsets_len = inflags.GetValueTensor("offsets_dims").lengths;
    SetTensorNd(offsetsTensor, offsets_len, int64_t_type);

    auto weight_len = inflags.GetValueTensor("weight_dims").lengths;
    SetTensorNd(weightTensor, weight_len, data_type);

    std::vector<int> out_len;
    if(in_len.size() == 2)
    {
        out_len.push_back(in_len[0]);
    }
    else if(in_len.size() == 1)
    {
        out_len.push_back(offsets_len[0]);
    }
    out_len.push_back(weight_len[1]);
    SetTensorNd(outputTensor, out_len, data_type);

    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> EmbeddingBagDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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
int EmbeddingBagDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward (1) (Default=1)", "int");
    inflags.AddTensorFlag("input_dims",
                          'I',
                          "1024x1024",
                          "The dimensional lengths of the input tensor (Default=1024x1024)");
    inflags.AddTensorFlag(
        "offsets_dims", 'O', "0", "The dimensional lengths of the offsets tensor (Default=0)");
    inflags.AddTensorFlag("weight_dims",
                          'W',
                          "1024x1024",
                          "The dimensional lengths of the weight tensor (Default=1024x1024)");
    inflags.AddTensorFlag("per_sample_weights_dims",
                          'P',
                          "0",
                          "The dimensional lengths of the per_sample_weights tensor (Default=0)");
    inflags.AddInputFlag(
        "mode",
        'M',
        "1",
        "Specifies the reduction to apply to the output ('sum (0)'|'mean (1)'|'max (2)') "
        "(Default=1)",
        "int");
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
int EmbeddingBagDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    size_t in_sz                 = GetTensorSpace(inputTensor);
    size_t weight_sz             = GetTensorSpace(weightTensor);
    size_t offsets_sz            = GetTensorSpace(offsetsTensor);
    size_t per_sample_weights_sz = GetTensorSpace(perSampleWeightsTensor);
    size_t out_sz                = GetTensorSpace(outputTensor);
    auto weight_len              = miopen::deref(weightTensor).GetLengths();

    if(forw == 1)
    {
        // GPU allocation
        input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(int64_t)));
        weight_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, weight_sz, sizeof(Tgpu)));
        offsets_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, offsets_sz, sizeof(int64_t)));
        per_sample_weights_dev =
            std::unique_ptr<GPUMem>(new GPUMem(ctx, per_sample_weights_sz, sizeof(Tgpu)));
        output_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

        // GPU host allocation
        input              = std::vector<int64_t>(in_sz, static_cast<int64_t>(0));
        weight             = std::vector<Tgpu>(weight_sz, static_cast<Tgpu>(0));
        offsets            = std::vector<int64_t>(offsets_sz, static_cast<int64_t>(0));
        per_sample_weights = std::vector<Tgpu>(per_sample_weights_sz, static_cast<Tgpu>(0));
        output             = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));

        // CPU allocation
        outputHost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

        for(int i = 0; i < in_sz; i++)
        {
            input[i] = prng::gen_A_to_B(static_cast<int64_t>(0.0),
                                        static_cast<int64_t>(weight_len[0] - 1));
        }
        for(int i = 0; i < weight_sz; i++)
        {
            weight[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        for(int i = 0; i < per_sample_weights_sz; i++)
        {
            per_sample_weights[i] =
                prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        for(int i = 0; i < offsets_sz; i++)
        {
            if(i == 0)
            {
                offsets[i] = prng::gen_A_to_B(static_cast<int64_t>(0.0),
                                              static_cast<int64_t>(weight_len[0] - 1));
            }
            else
            {
                offsets[i] = prng::gen_A_to_B(static_cast<int64_t>(offsets[i - 1]),
                                              static_cast<int64_t>(weight_len[0] - 1));
            }
        }

        if(input_dev->ToGPU(GetStream(), input.data()) != 0)
        {
            std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        if(weight_dev->ToGPU(GetStream(), weight.data()) != 0)
        {
            std::cerr << "Error copying (weight) to GPU, size: " << weight_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        if(offsets_dev->ToGPU(GetStream(), offsets.data()) != 0)
        {
            std::cerr << "Error copying (offsets) to GPU, size: " << offsets_dev->GetSize()
                      << std::endl;
            return miopenStatusInternalError;
        }
        if(per_sample_weights_dev->ToGPU(GetStream(), per_sample_weights.data()) != 0)
        {
            std::cerr << "Error copying (per_sample_weights) to GPU, size: "
                      << per_sample_weights_dev->GetSize() << std::endl;
            return miopenStatusInternalError;
        }
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int EmbeddingBagDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;
    Timer t;
    START_TIME;
    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenStatus_t status = miopenEmbeddingBagForward(GetHandle(),
                                                          inputTensor,
                                                          input_dev->GetMem(),
                                                          weightTensor,
                                                          weight_dev->GetMem(),
                                                          offsetsTensor,
                                                          offsets_dev->GetMem(),
                                                          perSampleWeightsTensor,
                                                          per_sample_weights_dev->GetMem(),
                                                          outputTensor,
                                                          output_dev->GetMem(),
                                                          mode);

        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenEmbeddingBagForward");

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
            std::cout << "Wall-clock Time Forward EmbeddingBag Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";
        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward EmbeddingBag Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int EmbeddingBagDriver<Tgpu, Tref>::RunForwardCPU()
{
    mloEmbeddingBagForward(inputTensor,
                           input.data(),
                           weightTensor,
                           weight.data(),
                           offsetsTensor,
                           offsets.data(),
                           perSampleWeightsTensor,
                           per_sample_weights.data(),
                           outputTensor,
                           outputHost.data(),
                           mode);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int EmbeddingBagDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
Tref EmbeddingBagDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int EmbeddingBagDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outputHost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward EmbeddingBag FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward EmbeddingBag OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int EmbeddingBagDriver<Tgpu, Tref>::RunBackwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref>
int EmbeddingBagDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
