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
#include "miopen/gatherv2/problem_description.hpp"
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

template <typename Tgpu, typename Tcheck, typename Tindex>
int32_t mloGatherV2BackwardRunHost(miopenTensorDescriptor_t outputGradDesc,
                                   Tgpu* outputGrad,
                                   miopenTensorDescriptor_t indicesDesc,
                                   Tindex* indices,
                                   miopenTensorDescriptor_t paramGradDesc,
                                   Tcheck* paramGrad,
                                   int64_t axis,
                                   int batch_dims)
{
    int64_t batch_size = 1;
    int64_t outer_size = 1;
    int64_t inner_size = 1;

    auto paramGrad_num_dim = miopen::deref(paramGradDesc).GetNumDims();
    auto paramGrad_lens    = miopen::deref(paramGradDesc).GetLengths();
    auto outGrad_numel     = miopen::deref(outputGradDesc).GetElementSize();
    auto indices_numel     = miopen::deref(indicesDesc).GetElementSize();

    for(int i = 0; i < batch_dims; i++)
    {
        batch_size *= paramGrad_lens[i];
    }
    for(int i = batch_dims; i < axis; i++)
    {
        outer_size *= paramGrad_lens[i];
    }
    for(int i = axis + 1; i < paramGrad_num_dim; i++)
    {
        inner_size *= paramGrad_lens[i];
    }

    int64_t gather_dim_size = paramGrad_lens[axis];

    if(batch_dims > 0)
    {
        auto outGrad_tv = miopen::gatherv2::reshape<4>(
            miopen::deref(outputGradDesc),
            {batch_size, outer_size, indices_numel / batch_size, inner_size});
        auto paramGrad_tv = miopen::gatherv2::reshape<4>(
            miopen::deref(paramGradDesc), {batch_size, outer_size, gather_dim_size, inner_size});

        const bool is_batch_dims_zero = (batch_size == 1);
        const bool is_axis_zero       = (outer_size == 1);

        for(long i = 0; i < outGrad_numel; i++)
        {
            long batch_i   = 0;
            long outer_i   = 0;
            long indices_i = 0;
            long slice_i   = 0;

            const long slices_count = i / inner_size;
            if(is_batch_dims_zero)
            {
                if(is_axis_zero)
                {
                    indices_i = slices_count;
                }
                else
                {
                    outer_i   = slices_count / gather_dim_size;
                    indices_i = slices_count - outer_i * gather_dim_size;
                }
            }
            else
            {
                const long entries_count = slices_count / gather_dim_size;
                if(is_axis_zero)
                {
                    batch_i = entries_count;
                }
                else
                {
                    batch_i = entries_count / outer_size;
                    outer_i = entries_count - batch_i * outer_size;
                }
                indices_i = slices_count - entries_count * inner_size;
            }
            slice_i = i - slices_count * inner_size;

            size_t gather_i = indices[batch_i * gather_dim_size + indices_i];

            if(gather_i < gather_dim_size)
            {
                long param_i =
                    ((batch_i * outer_size + outer_i) * gather_dim_size) * inner_size + slice_i;
                paramGrad[param_i] += getNDVal(outputGrad, outGrad_tv, i);
            }
        }
    }
    else
    {
        auto outputGrad_tv = miopen::gatherv2::reshape<3>(miopen::deref(outputGradDesc),
                                                          {outer_size, indices_numel, inner_size});
        bool is_axis_zero  = (outer_size == 1);

        for(long i = 0; i < outGrad_numel; i++)
        {
            long batch_i   = 0;
            long indices_i = 0;
            long slice_i   = 0;
            if(is_axis_zero)
            {
                indices_i = i / inner_size;
                slice_i   = i - indices_i * inner_size;
            }
            else
            {
                long batch_indices_i = i / inner_size;
                batch_i              = batch_indices_i / indices_numel;
                indices_i            = batch_indices_i - batch_i * indices_numel;
                slice_i              = i - batch_indices_i * inner_size;
            }

            size_t gather_i = indices[indices_i];

            if(gather_i < gather_dim_size)
            {
                long param_i = (batch_i * gather_dim_size + gather_i) * inner_size + slice_i;
                paramGrad[param_i] += getNDVal(outputGrad, outputGrad_tv, i);
            }
        }
    }

    return 0;
}

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

        data_type       = miopen_type<Tgpu>{};
        index_data_type = miopen_type<Tindex>{};
    }

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
    miopenDataType_t index_data_type;

    int forw;

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
    axis       = inflags.GetValueInt("axis");
    batch_dims = inflags.GetValueInt("batch-dims");
    forw       = inflags.GetValueInt("forw");
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
    SetTensorNd(paramGradTensor, paramGrad_len, data_type);

    std::vector<int> indices_len = inflags.GetValueTensor("indices-shape").lengths;
    SetTensorNd(indicesTensor, indices_len, index_data_type);

    std::vector<int> outGrad_len;
    for(int i = 0; i < axis; i++)
    {
        outGrad_len.push_back(paramGrad_len[i]);
    }

    for(int i = batch_dims; i < indices_len.size(); i++)
    {
        outGrad_len.push_back(indices_len[i]);
    }

    for(int i = axis + 1; i < paramGrad_len.size(); i++)
    {
        outGrad_len.push_back(paramGrad_len[i]);
    }

    SetTensorNd(outputGradTensor, outGrad_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run only Forward (1) or Run both Forward and Backward (0) (Default = 0)",
                         "int");
    inflags.AddTensorFlag("paramGrad-shape", 'P', "2x4", "The shape of the param gradient tensor");
    inflags.AddTensorFlag("indices-shape", '4', "1", "The shape of the indices tensor");
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
        size_t paramGrad_sz = GetTensorSpace(paramGradTensor);
        size_t outGrad_sz   = GetTensorSpace(outputGradTensor);
        size_t indices_sz   = GetTensorSpace(indicesTensor);

        // GPU allocation
        paramGrad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, paramGrad_sz, sizeof(Tgpu)));
        outputGrad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, outGrad_sz, sizeof(Tgpu)));
        indices_dev    = std::unique_ptr<GPUMem>(new GPUMem(ctx, indices_sz, sizeof(Tindex)));

        // GPU host allocation
        paramGrad = std::vector<Tgpu>(paramGrad_sz, static_cast<Tgpu>(0));
        outGrad   = std::vector<Tgpu>(outGrad_sz, static_cast<Tgpu>(0));
        indices   = std::vector<Tindex>(indices_sz, static_cast<Tindex>(0));

        // CPU allocation
        paramGradHost = std::vector<Tref>(paramGrad_sz, static_cast<Tref>(0));

        for(int i = 0; i < outGrad_sz; i++)
        {
            outGrad[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }

        if(indices_dev->ToGPU(GetStream(), indices.data()) != 0)
            std::cerr << "Error copying (indices) to GPU, size: " << indices_dev->GetSize()
                      << std::endl;

        if(outputGrad_dev->ToGPU(GetStream(), outGrad.data()) != 0)
            std::cerr << "Error copying (outputGrad) to GPU, size: " << outputGrad_dev->GetSize()
                      << std::endl;
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
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenGatherV2Backward(GetHandle(),
                               outputGradTensor,
                               outputGrad_dev->GetMem(),
                               indicesTensor,
                               indices_dev->GetMem(),
                               paramGradTensor,
                               paramGrad_dev->GetMem(),
                               axis,
                               batch_dims);
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
            std::cout << "Wall-clock Time Backward GatherV2 Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward GatherV2 Elapsed: " << kernel_average_time
                  << " ms\n";
    }

    if(paramGrad_dev->FromGPU(GetStream(), paramGrad.data()) != 0)
        std::cerr << "Error copying (paramGrad_dev) from GPU, size: " << paramGrad_dev->GetSize()
                  << std::endl;

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
    mloGatherV2BackwardRunHost<Tgpu, Tref, Tindex>(outputGradTensor,
                                                   outGrad.data(),
                                                   indicesTensor,
                                                   indices.data(),
                                                   paramGradTensor,
                                                   paramGradHost.data(),
                                                   axis,
                                                   batch_dims);
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherV2Driver<Tgpu, Tref, Tindex>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(paramGradHost, paramGrad);
    for(int i = 0; i < paramGradHost.size(); i++)
    {
        if(paramGradHost[i] != paramGrad[i])
        {
            std::cout << "paramGradHost[" << i << "] = " << paramGradHost[i] << " != "
                      << "paramGrad[" << i << "] = " << paramGrad[i] << std::endl;
        }
    }

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward GatherV2 FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward GatherV2 Verifies OK on CPU reference (" << error << " < "
                  << tolerance << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_GATHERV2_DRIVER_HPP