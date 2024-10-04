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
#include "miopen/gather/problem_description.hpp"
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
        printf("indices numel: %ld\n", indices_numel);
        printf("batch_size: %ld\n", batch_size);
        auto outGrad_tv = miopen::gather::reshape<4>(
            miopen::deref(outputGradDesc),
            {batch_size, outer_size, indices_numel / batch_size, inner_size});

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
        auto outputGrad_tv = miopen::gather::reshape<3>(miopen::deref(outputGradDesc),
                                                        {outer_size, indices_numel, inner_size});
        bool is_axis_zero  = (outer_size == 1);

        for(long i = 0; i < outGrad_numel; i++)
        {
            long outer_i   = 0;
            long indices_i = 0;
            long inner_i   = 0;
            if(is_axis_zero)
            {
                indices_i = i / inner_size;
                inner_i   = i - indices_i * inner_size;
            }
            else
            {
                long batch_indices_i = i / inner_size;
                outer_i              = batch_indices_i / indices_numel;
                indices_i            = batch_indices_i - outer_i * indices_numel;
                inner_i              = i - batch_indices_i * inner_size;
            }

            size_t gather_i = indices[indices_i];

            if(gather_i < gather_dim_size)
            {
                long param_i = (outer_i * gather_dim_size + gather_i) * inner_size + inner_i;
                paramGrad[param_i] += getNDVal(outputGrad, outputGrad_tv, i);
                // paramGrad[outer_i][gather_i][inner_i] += outputGrad[i];
                // paramGrad[param_i] += outputGrad[i];
            }
        }
    }

    return 0;
}

template <typename Tgpu, typename Tref, typename Tindex>
class GatherDriver : public Driver
{
public:
    GatherDriver() : Driver()
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
    ~GatherDriver() override
    {
        miopenDestroyTensorDescriptor(outputGradTensor);
        miopenDestroyTensorDescriptor(indicesTensor);
        miopenDestroyTensorDescriptor(paramGradTensor);
    }

private:
    int forw;

    InputFlags inflags;
    miopenDataType_t index_data_type;
    miopenGatherDescriptor_t gatherDesc;

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

    miopenGatherMode_t mode;
    uint32_t dim;
    uint32_t batch_dims;
};

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    forw = inflags.GetValueInt("forw");
    MIOPEN_THROW_IF(forw != 0, "Incorrect Forward Mode");

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    dim        = inflags.GetValueInt("dim");
    batch_dims = inflags.GetValueInt("batch_dims");
    if(inflags.GetValueStr("mode") == "gather")
    {
        mode = MIOPEN_GATHER;
    }
    else if(inflags.GetValueStr("mode") == "gatherv2")
    {
        mode = MIOPEN_GATHER_V2;
    }
    else if(inflags.GetValueStr("mode") == "gathernd")
    {
        mode = MIOPEN_GATHER_ND;
    }
    else
    {
        MIOPEN_THROW("Incorrect Gather Mode");
    }

    miopenCreateGatherDescriptor(&gatherDesc);
    miopenSetGatherDescriptor(gatherDesc, mode, dim, batch_dims);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::GetandSetData()
{
    std::vector<int> paramGrad_len = inflags.GetValueTensor("param_grad_shape").lengths;
    SetTensorNd(paramGradTensor, paramGrad_len, data_type);

    std::vector<int> indices_len = inflags.GetValueTensor("indices_shape").lengths;
    SetTensorNd(indicesTensor, indices_len, index_data_type);

    std::vector<int> outGrad_len;
    for(int i = 0; i < dim; i++)
    {
        outGrad_len.push_back(paramGrad_len[i]);
    }

    for(int i = batch_dims; i < indices_len.size(); i++)
    {
        outGrad_len.push_back(indices_len[i]);
    }

    for(int i = dim + 1; i < paramGrad_len.size(); i++)
    {
        outGrad_len.push_back(paramGrad_len[i]);
    }

    SetTensorNd(outputGradTensor, outGrad_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run only Forward (1) or Run both Forward and Backward (0) (Default = 0)",
                         "int");
    inflags.AddTensorFlag("param_grad_shape", 'P', "2x4", "The shape of the param gradient tensor");
    inflags.AddTensorFlag("indices_shape", 'I', "3", "The shape of the indices tensor");
    inflags.AddInputFlag("mode",
                         'm',
                         "gatherv2",
                         "Gather Mode (gather, gatherv2, gathernd) (Default=gatherv2)",
                         "str");
    inflags.AddInputFlag(
        "dim", 'D', "0", "The dimension in params to gather indices from (Default=0)", "int");
    inflags.AddInputFlag(
        "batch_dims", 'B', "0", "The number of batch dimensions (Default=0)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::AllocateBuffersAndCopy()
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

        for(int i = 0; i < indices_sz; i++)
        {
            // indices[i] = prng::gen_A_to_B(static_cast<Tindex>(0), static_cast<Tindex>(1));
            indices[i] = 0;
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
int GatherDriver<Tgpu, Tref, Tindex>::RunForwardGPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::RunForwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenGatherBackward(GetHandle(),
                             gatherDesc,
                             outputGradTensor,
                             outputGrad_dev->GetMem(),
                             indicesTensor,
                             indices_dev->GetMem(),
                             paramGradTensor,
                             paramGrad_dev->GetMem(),
                             &dim,
                             &batch_dims);

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
Tref GatherDriver<Tgpu, Tref, Tindex>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::VerifyForward()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::RunBackwardCPU()
{
    if(mode == MIOPEN_GATHER_V2)
    {
        mloGatherV2BackwardRunHost<Tgpu, Tref, Tindex>(outputGradTensor,
                                                       outGrad.data(),
                                                       indicesTensor,
                                                       indices.data(),
                                                       paramGradTensor,
                                                       paramGradHost.data(),
                                                       dim,
                                                       batch_dims);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::VerifyBackward()
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
        std::cout << "Backward Gather FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward Gather Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
