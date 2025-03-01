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

#include "gtest_common.hpp"

Gpu GetDevGpuType()
{
    const auto dev_name = get_handle().GetDeviceName();

    static const auto dev = [&] {
        if(dev_name == "gfx900")
            return Gpu::gfx900;
        else if(dev_name == "gfx906")
            return Gpu::gfx906;
        else if(dev_name == "gfx908")
            return Gpu::gfx908;
        else if(dev_name == "gfx90a")
            return Gpu::gfx90A;
        else if(miopen::StartsWith(dev_name, "gfx94"))
            return Gpu::gfx94X;
        else if(miopen::StartsWith(dev_name, "gfx103"))
            return Gpu::gfx103X;
        else if(miopen::StartsWith(dev_name, "gfx110"))
            return Gpu::gfx110X;
        else
            throw std::runtime_error("unknown_gpu");
    }();

    return dev;
}

bool IsTestSupportedByDevice(Gpu supported_devs)
{
    if((supported_devs & GetDevGpuType()) != Gpu::None)
    {
        return true;
    }
    return false;
}
