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
#ifndef GUARD_MIOPEN_VAR_DRIVER_HPP
#define GUARD_MIOPEN_VAR_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "random.hpp"
#include "../test/verify.hpp"
#include "mloVarHost.hpp"

#include <algorithm>
#include <cstdlib>
#include <cfloat>
#include <memory>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>

template <typename Tgpu, typename Tref>
class VarDriver : public Driver
{
public:
    VarDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);
        miopenCreateTensorDescriptor(&meanDesc);
        miopenCreateTensorDescriptor(&meanGradDesc);
        miopenCreateTensorDescriptor(&varGradDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetInputTensorLengthsFromCmdLine();
    std::vector<int> GetDimsFromCmdLine();
    bool GetKeepDimFromCmdLine();
    bool GetUnbiasedFromCmdLine();
    int GetDivisorFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();
    int VerifyForward() override;
    int VerifyBackward() override;

    ~VarDriver()
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
        miopenDestroyTensorDescriptor(meanDesc);
        miopenDestroyTensorDescriptor(meanGradDesc);
        miopenDestroyTensorDescriptor(varGradDesc);
    }

private:
    InputFlags inflags;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t inputGradDesc;
    miopenTensorDescriptor_t meanDesc;
    miopenTensorDescriptor_t meanGradDesc;
    miopenTensorDescriptor_t varGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> input_grad_dev;
    std::unique_ptr<GPUMem> mean_dev;
    std::unique_ptr<GPUMem> mean_grad_dev;
    std::unique_ptr<GPUMem> var_grad_dev;

    std::vector<Tgpu> input;
    std::vector<Tgpu> input_grad;
    std::vector<Tgpu> mean;
    std::vector<Tgpu> mean_grad;
    std::vector<Tgpu> var_grad;

    std::vector<Tref> input_grad_host;

    std::vector<int> dims;

    bool keepdim;
    bool unbiased;

    int32_t divisor;
};

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> input_len = GetInputTensorLengthsFromCmdLine();
    std::vector<int> input_grad_len(input_len.size());
    std::vector<int> mean_len(input_len.size());
    std::vector<int> mean_grad_len(input_len.size());
    std::vector<int> var_grad_len(input_len.size());

    std::copy(input_len.begin(), input_len.end(), input_grad_len.begin());
    std::copy(input_len.begin(), input_len.end(), mean_len.begin());
    std::copy(input_len.begin(), input_len.end(), mean_grad_len.begin());
    std::copy(input_len.begin(), input_len.end(), var_grad_len.begin());

    dims     = GetDimsFromCmdLine();
    keepdim  = GetKeepDimFromCmdLine();
    unbiased = GetUnbiasedFromCmdLine();
    divisor  = GetDivisorFromCmdLine();

    for(const auto& dim : dims)
    {
        mean_len[dim]      = 1;
        mean_grad_len[dim] = 1;
        var_grad_len[dim]  = 1;
    }

    SetTensorNd(inputDesc, input_len, data_type);
    SetTensorNd(inputGradDesc, input_grad_len, data_type);
    SetTensorNd(meanDesc, mean_len, data_type);
    SetTensorNd(meanGradDesc, mean_grad_len, data_type);
    SetTensorNd(varGradDesc, var_grad_len, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "0", "Run only forward pass (Default=0)", "int");
    inflags.AddInputFlag("InputDims",
                         'I',
                         "32,32,32",
                         "The dimensional lengths of the input tensor (Default=32x32x32)",
                         "string");
    inflags.AddInputFlag("Dims", 'D', "0", "The dimensions to reduce (Default=0)", "string");
    inflags.AddInputFlag("KeepDim", 'K', "1", "Keep the reduced dimensions (Default=1)", "int");
    inflags.AddInputFlag("Unbiased", 'U', "1", "Use unbiased variance (Default=1)", "int");
    inflags.AddInputFlag("Divisor", 'V', "32", "The divisor to use (Default=32)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'v', "1", "Verify the results (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time flag (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<int> VarDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    std::string input_dims_str = inflags.GetValueStr("InputDims");

    std::vector<int> input_dims;
    size_t pos = 0;
    size_t new_pos;

    new_pos = input_dims_str.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = input_dims_str.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        input_dims.push_back(len);

        pos     = new_pos + 1;
        new_pos = input_dims_str.find(',', pos);
    };

    std::string sliceStr = input_dims_str.substr(pos);
    int len              = std::stoi(sliceStr);

    input_dims.push_back(len);

    return (input_dims);
}

template <typename Tgpu, typename Tref>
std::vector<int> VarDriver<Tgpu, Tref>::GetDimsFromCmdLine()
{
    std::string dims_str = inflags.GetValueStr("Dims");

    std::vector<int> dims_;
    size_t pos = 0;
    size_t new_pos;

    new_pos = dims_str.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = dims_str.substr(pos, new_pos - pos);

        int dim = std::stoi(sliceStr);

        dims_.push_back(dim);

        pos     = new_pos + 1;
        new_pos = dims_str.find(',', pos);
    };

    std::string sliceStr = dims_str.substr(pos);
    int dim              = std::stoi(sliceStr);

    dims_.push_back(dim);

    return (dims_);
}

template <typename Tgpu, typename Tref>
bool VarDriver<Tgpu, Tref>::GetKeepDimFromCmdLine()
{
    int keepdim_ = inflags.GetValueInt("KeepDim");

    return (keepdim_ == 1 ? true : false);
}

template <typename Tgpu, typename Tref>
bool VarDriver<Tgpu, Tref>::GetUnbiasedFromCmdLine()
{
    int unbiased_ = inflags.GetValueInt("Unbiased");

    return (unbiased_ == 1 ? true : false);
}

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::GetDivisorFromCmdLine()
{
    return (inflags.GetValueInt("Divisor"));
}

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    size_t input_sz      = GetTensorSize(inputDesc);
    size_t input_grad_sz = GetTensorSize(inputGradDesc);
    size_t mean_sz       = GetTensorSize(meanDesc);
    size_t mean_grad_sz  = GetTensorSize(meanGradDesc);
    size_t var_grad_sz   = GetTensorSize(varGradDesc);

    uint32_t ctx = 0;

    input_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(Tgpu)));
    input_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_grad_sz, sizeof(Tgpu)));
    mean_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, mean_sz, sizeof(Tgpu)));
    mean_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, mean_grad_sz, sizeof(Tgpu)));
    var_grad_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, var_grad_sz, sizeof(Tgpu)));

    input           = std::vector<Tgpu>(input_sz, static_cast<Tgpu>(0));
    input_grad      = std::vector<Tgpu>(input_grad_sz, static_cast<Tgpu>(0));
    input_grad_host = std::vector<Tref>(input_grad_sz, static_cast<Tref>(0));
    mean            = std::vector<Tgpu>(mean_sz, static_cast<Tgpu>(0));
    mean_grad       = std::vector<Tgpu>(mean_grad_sz, static_cast<Tgpu>(0));
    var_grad        = std::vector<Tgpu>(var_grad_sz, static_cast<Tgpu>(0));

    int status;

    for(int i = 0; i < input_sz; i++)
    {
        input[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status = input_dev->ToGPU(q, input.data());

    for(int i = 0; i < input_grad_sz; i++)
    {
        input_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= input_grad_dev->ToGPU(q, input_grad.data());

    mloMeanForwardRunHost(inputDesc, meanDesc, input.data(), mean.data(), dims, divisor);
    status |= mean_dev->ToGPU(q, mean.data());

    for(int i = 0; i < mean_grad_sz; i++)
    {
        mean_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= mean_grad_dev->ToGPU(q, mean_grad.data());

    for(int i = 0; i < var_grad_sz; i++)
    {
        var_grad[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
    }
    status |= var_grad_dev->ToGPU(q, var_grad.data());

    if(status != 0)
    {
        std::cout << "Error copying data to GPU\n" << std::endl;

        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::RunForwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::RunForwardCPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenVarBackward(GetHandle(),
                          inputDesc,
                          input_dev->GetMem(),
                          inputGradDesc,
                          input_grad_dev->GetMem(),
                          meanDesc,
                          mean_dev->GetMem(),
                          meanGradDesc,
                          mean_grad_dev->GetMem(),
                          varGradDesc,
                          var_grad_dev->GetMem(),
                          dims.data(),
                          dims.size(),
                          keepdim,
                          unbiased,
                          divisor);
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
            printf("Wall-clock Time Backward Var Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Backward Var Elapsed: %f ms\n", kernel_average_time);
    }

    input_grad_dev->FromGPU(GetStream(), input_grad.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::RunBackwardCPU()
{
    mloVarBackwardRunHost<Tgpu, Tref>(inputDesc,
                                      inputGradDesc,
                                      meanDesc,
                                      meanGradDesc,
                                      varGradDesc,
                                      input.data(),
                                      input_grad_host.data(),
                                      mean.data(),
                                      mean_grad.data(),
                                      var_grad.data(),
                                      dims.data(),
                                      dims.size(),
                                      unbiased,
                                      divisor);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref VarDriver<Tgpu, Tref>::GetTolerance()
{
    auto tolerance = std::is_same<Tgpu, float>::value ? 1.5e-6 : 8.2e-3;

    if(std::is_same<Tgpu, bfloat16>::value)
        tolerance *= 8.0;

    return tolerance;
}

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::VerifyForward()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int VarDriver<Tgpu, Tref>::VerifyBackward()
{
    RunBackwardCPU();
    const Tref tolerance = GetTolerance();
    auto error           = miopen::rms_range(input_grad_host, input_grad);

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward Var Failed: " << error << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        printf("Backward Var Verifies on CPU and GPU (err=%f)\n", error);
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_VAR_DRIVER_HPP
