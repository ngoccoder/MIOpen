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

#include "var.hpp"
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace env = miopen::env;

namespace var {

std::string GetFloatArg()
{
    const auto tmp = env::value(MIOPEN_TEST_FLOAT_ARG);
    if(tmp.empty())
        return "";
    return tmp;
}

struct VarBackwardTestContiguousFloat : VarBackwardTestContiguous<float>
{
};

struct VarBackwardTestContiguousHalf : VarBackwardTestContiguous<half_float::half>
{
};

struct VarBackwardTestContiguousBFloat16 : VarBackwardTestContiguous<bfloat16>
{
};

struct VarBackwardTestNonContiguousFloat : VarBackwardTestNonContiguous<float>
{
};

struct VarBackwardTestNonContiguousHalf : VarBackwardTestNonContiguous<half_float::half>
{
};

struct VarBackwardTestNonContiguousBFloat16 : VarBackwardTestNonContiguous<bfloat16>
{
};

} // namespace var
using namespace var;

TEST_P(VarBackwardTestContiguousFloat, VarTestBw)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--float")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(VarBackwardTestContiguousHalf, VarTestBw)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--half")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(VarBackwardTestContiguousBFloat16, VarTestBw)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--bfloat16")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(VarBackwardTestNonContiguousFloat, VarTestBw)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--float")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(VarBackwardTestNonContiguousHalf, VarTestBw)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--half")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

TEST_P(VarBackwardTestNonContiguousBFloat16, VarTestBw)
{
    if(!MIOPEN_TEST_ALL || (env::enabled(MIOPEN_TEST_ALL) && (GetFloatArg() == "--bfloat16")))
    {
        RunTest();
        Verify();
    }
    else
    {
        GTEST_SKIP();
    }
}

INSTANTIATE_TEST_SUITE_P(VarTestSet,
                         VarBackwardTestContiguousFloat,
                         testing::ValuesIn(VarTestConfigs()));
INSTANTIATE_TEST_SUITE_P(VarTestSet,
                         VarBackwardTestContiguousHalf,
                         testing::ValuesIn(VarTestConfigs()));
INSTANTIATE_TEST_SUITE_P(VarTestSet,
                         VarBackwardTestContiguousBFloat16,
                         testing::ValuesIn(VarTestConfigs()));
INSTANTIATE_TEST_SUITE_P(VarTestSet,
                         VarBackwardTestNonContiguousFloat,
                         testing::ValuesIn(VarTestConfigs()));
INSTANTIATE_TEST_SUITE_P(VarTestSet,
                         VarBackwardTestNonContiguousHalf,
                         testing::ValuesIn(VarTestConfigs()));
INSTANTIATE_TEST_SUITE_P(VarTestSet,
                         VarBackwardTestNonContiguousBFloat16,
                         testing::ValuesIn(VarTestConfigs()));
