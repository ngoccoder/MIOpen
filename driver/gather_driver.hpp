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
#include "mloGatherHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/verify.hpp>

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

#include <memory>
#include <vector>

template <typename Tgpu, typename Tref, typename Tindex>
class GatherDriver : public Driver
{
public:
    GatherDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&indicesTensor);
        miopenCreateTensorDescriptor(&outputTensor);

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
        miopenDestroyTensorDescriptor(inputTensor);
        miopenDestroyTensorDescriptor(indicesTensor);
        miopenDestroyTensorDescriptor(outputTensor);
    }

private:
    int forw;

    InputFlags inflags;
    miopenDataType_t index_data_type;
    miopenGatherDescriptor_t gatherDesc;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t indicesTensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> indices_dev;
    std::unique_ptr<GPUMem> output_dev;

    std::vector<Tgpu> input;
    std::vector<Tindex> indices;
    std::vector<Tgpu> output;

    std::vector<Tref> outputHost;

    miopenGatherMode_t mode;
    uint32_t dim;
    uint32_t batch_dims;
};

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    forw = inflags.GetValueInt("forw");
    MIOPEN_THROW_IF(forw != 0 && forw != 1, "Incorrect Forward Mode");

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
    std::vector<int> input_len = inflags.GetValueTensor("input_shape").lengths;
    SetTensorNd(inputTensor, input_len, data_type);

    std::vector<int> indices_len = inflags.GetValueTensor("indices_shape").lengths;
    SetTensorNd(indicesTensor, indices_len, index_data_type);

    std::vector<int> output_len;

    // output shape = param[:axis] + indice[batch_dim:] + param[axis + 1:]
    if(mode == MIOPEN_GATHER)
    {
        output_len = indices_len;
    }
    else
    {
        return miopenStatusNotImplemented;
    }

    SetTensorNd(outputTensor, output_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run both Forward and Backward (0), Run only Forward (1) (Default = 1)",
                         "int");
    inflags.AddTensorFlag(
        "input_shape", 'P', "2x3x5x5", "The shape of the input tensor (Default = 2x3x5x5)");
    inflags.AddTensorFlag(
        "indices_shape", 'I', "2x3x3x3", "The shape of the indices tensor (Default = 2x3x3x3)");
    inflags.AddInputFlag(
        "mode", 'm', "gather", "Gather Mode (gather, gatherv2, gathernd) (Default=gather)", "str");
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

    size_t input_sz   = GetTensorSpace(inputTensor);
    size_t output_sz  = GetTensorSpace(outputTensor);
    size_t indices_sz = GetTensorSpace(indicesTensor);

    // GPU allocation
    input_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    output_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, output_sz, sizeof(Tgpu)));
    indices_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, indices_sz, sizeof(Tindex)));

    // GPU host allocation
    input   = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    output  = std::vector<Tgpu>(output_sz, static_cast<Tgpu>(0));
    indices = std::vector<Tindex>(indices_sz, static_cast<Tindex>(0));

    // CPU allocation
    outputHost = std::vector<Tref>(output_sz, static_cast<Tref>(0));

    for(size_t i = 0; i < input_sz; i++)
    {
        input[i] = prng::gen_A_to_B(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }

    for(size_t i = 0; i < indices_sz; i++)
    {
        auto input_shape = miopen::deref(inputTensor).GetLengths();
        indices[i] =
            prng::gen_A_to_B(static_cast<Tindex>(0), static_cast<Tindex>(input_shape[dim]));
    }

    if(indices_dev->ToGPU(GetStream(), indices.data()) != 0)
    {
        std::cerr << "Error copying (indices) to GPU, size: " << indices_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        auto status = miopenGatherForward(GetHandle(),
                                          gatherDesc,
                                          inputTensor,
                                          input_dev->GetMem(),
                                          indicesTensor,
                                          indices_dev->GetMem(),
                                          outputTensor,
                                          output_dev->GetMem());
        MIOPEN_THROW_IF(status != miopenStatusSuccess, "Error in miopenGatherForward");

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
            std::cout << "Wall-clock Time Forward Gather Elapsed: " << t.gettime_ms() / iter
                      << " ms\n";

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward Gather Elapsed: " << kernel_average_time << " ms\n";
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::RunForwardCPU()
{
    int status = miopenStatusSuccess;
    if(mode == MIOPEN_GATHER)
    {
        status = mloGatherForwardRunHost<Tgpu, Tref, Tindex>(inputTensor,
                                                             input.data(),
                                                             indicesTensor,
                                                             indices.data(),
                                                             outputTensor,
                                                             outputHost.data(),
                                                             dim);
    }
    else
    {
        return miopenStatusNotImplemented;
    }

    return status;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::RunBackwardGPU()
{
    return miopenStatusNotImplemented;
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
    RunForwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(outputHost, output);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Gather FAILED: " << error << " > " << tolerance << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Gather Verifies OK on CPU reference (" << error << " < " << tolerance
                  << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::RunBackwardCPU()
{
    return miopenStatusNotImplemented;
}

template <typename Tgpu, typename Tref, typename Tindex>
int GatherDriver<Tgpu, Tref, Tindex>::VerifyBackward()
{
    return miopenStatusNotImplemented;
}
