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
#ifndef GUARD_MIOPEN_GATHERV2_DRIVER_HPP
#define GUARD_MIOPEN_GATHERV2_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "random.hpp"
#include "tensor_view.hpp"
#include "timer.hpp"

#include <../test/verify.hpp>

#include <memory>
#include <miopen/errors.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <miopen/miopen.h>

#include <vector>

#ifndef MLO_GATHERV2HOST_H_
#define MLO_GATHERV2HOST_H_

/*
template <typename Tgpu, typename Tcheck>
int32_t mloDiagForwardRunHost(miopenTensorDescriptor_t inputDesc,
                              Tgpu* input,
                              miopenTensorDescriptor_t outputDesc,
                              Tcheck* outputHost,
                              int64_t diagonal)
{
    auto in_len = miopen::deref(inputDesc).GetLengths();
    if(in_len.size() == 1)
    {
        auto input_numel = miopen::deref(inputDesc).GetElementSize();
        auto output_tv   = miopen::get_inner_expanded_tv<2>(miopen::deref(outputDesc));
        auto offset =
            (diagonal >= 0) ? diagonal * output_tv.stride[1] : -diagonal * output_tv.stride[0];

        for(size_t i = 0; i < input_numel; i++)
        {
            long outputIdx        = i * (output_tv.stride[0] + output_tv.stride[1]) + offset;
            outputHost[outputIdx] = input[i];
        }
    }
    else if(in_len.size() == 2)
    {
        auto output_numel = miopen::deref(outputDesc).GetElementSize();
        auto input_tv     = miopen::get_inner_expanded_tv<2>(miopen::deref(inputDesc));
        auto output_tv    = miopen::get_inner_expanded_tv<1>(miopen::deref(outputDesc));
        auto offset =
            (diagonal >= 0) ? diagonal * input_tv.stride[1] : -diagonal * input_tv.stride[0];

        for(size_t i = 0; i < output_numel; i++)
        {
            long inputIdx = i * (input_tv.stride[0] + input_tv.stride[1]) + offset;
            Tcheck val    = static_cast<Tcheck>(getNDVal(input, input_tv, inputIdx));
            setNDVal(outputHost, output_tv, i, val);
        }
    }

    return 0;
}
*/

#endif

template <typename Tgpu, typename Tref, typename Tindex>
class GatherV2Driver : public Driver
{
public:
    GatherV2Driver() : Driver()
    {
        miopenCreateTensorDescriptor(&outputGradTensor);
        miopenCreateTensorDescriptor(&indicesTensor);
        miopenCreateTensorDescriptor(&paramGradTensor);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> inputDim);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU(); // Verify implements it

    int RunBackwardGPU() override;
    int RunBackwardCPU(); // Verify implements it

    Tref GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~GatherV2Driver() override
    {
        miopenDestroyTensorDescriptor(outputGradTensor);
        miopenDestroyTensorDescriptor(indicesTensor);
        miopenDestroyTensorDescriptor(paramGradTensor);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

    miopenTensorDescriptor_t outputGradTensor;
    miopenTensorDescriptor_t indicesTensor;
    miopenTensorDescriptor_t paramGradTensor;

    std::unique_ptr<GPUMem> outputGrad_dev;
    std::unique_ptr<GPUMem> indices_dev;
    std::unique_ptr<GPUMem> paramGrad_dev;

    std::vector<Tgpu> outGrad;
    std::vector<Tindex> indices;
    std::vector<Tgpu> paramGrad;
    std::vector<Tref> paramGradHost;

    int64_t axis;
    int batch_dims;
};

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);
    axis         = inflags.GetValueInt("axis");
    isContiguous = inflags.GetValueInt("contiguous") > 0 ? true : false;

    forw = inflags.GetValueInt("forw");
    if(forw != 0)
    {
        printf("Incorrect Forward Mode\n");
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::GetandSetData()
{
    std::vector<int> paramGrad_len = inflags.GetValueTensor("paramGrad-shape").lengths;
    auto paramGradStride           = ComputeStrides(paramGrad_len);
    SetTensorNd(paramGradTensor, paramGrad_len, paramGradStride, data_type);

    std::vector<int> indices_len = inflags.GetValueTensor("indices-shape").lengths;
    auto indicesStride           = ComputeStrides(indices_len);
    SetTensorNd(indicesTensor, indices_len, indicesStride, data_type);

    std::vector<int> outGrad_len;

    if(in_len.size() == 1)
    {
        size_t sz = in_len[0] + abs(diagonal);
        out_len   = {sz, sz};
    }
    else if(in_len.size() == 2)
    {
        int64_t sz = 0;
        if(diagonal >= 0)
        {
            sz = std::min(static_cast<int64_t>(in_len[0]), in_len[1] - diagonal);
        }
        else
        {
            sz = std::min(in_len[0] + diagonal, static_cast<int64_t>(in_len[1]));
        }

        if(sz <= 0)
        {
            isOutputRequired = false;
        }
        else
        {
            out_len = {sz};
        }
    }

    SetTensorNd(outputTensor, out_len, data_type);

    return miopenStatusSuccess;
}

// Equivalent tensor.transpose(0, -1).contiguous().transpose(0, -1)
template <typename Tgpu, typename Tref>
std::vector<int> DiagDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
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

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run only Forward (1) or Run both Forward and Backward (0) (Default = 0)",
                         "int");
    inflags.AddTensorFlag(
        "paramGrad-shape", 'P', "256x512", "The shape of the param gradient tensor");
    inflags.AddTensorFlag("indices-shape", 'I', "256", "The shape of the indices tensor");
    inflags.AddInputFlag(
        "axis", 'A', "0", "The axis in params to gather indices from (Default=0)", "int");
    inflags.AddInputFlag(
        "batch-dims", 'B', "0", "The number of batch dimensions (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;

    if(forw == 0)
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

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::RunForwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::RunForwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
Tref GatherV2Driver<Tgpu, Tref, Tindex>::GetTolerance()
{
    // Computation error of fp16 is ~2^13 (=8192) bigger than
    // the one of fp32 because mantissa is shorter by 13 bits.
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::VerifyForward()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::RunBackwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_GATHERV2_DRIVER_HPP