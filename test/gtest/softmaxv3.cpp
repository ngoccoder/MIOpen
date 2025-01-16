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

#include "softmaxv3.hpp"
using float16 = half_float::half;

namespace softmax {

using GPU_Softmax_fwd_FP32  = SoftmaxFwdTest<float>;
using GPU_Softmax_fwd_FP16  = SoftmaxFwdTest<float16>;
using GPU_Softmax_fwd_BFP16 = SoftmaxFwdTest<bfloat16>;

using GPU_Softmax_bwd_FP32  = SoftmaxBwdTest<float>;
using GPU_Softmax_bwd_FP16  = SoftmaxBwdTest<float16>;
using GPU_Softmax_bwd_BFP16 = SoftmaxBwdTest<bfloat16>;

} // namespace softmax
using namespace softmax;

TEST_P(GPU_Softmax_fwd_FP32, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Softmax_fwd_FP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Softmax_fwd_BFP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Softmax_bwd_FP32, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Softmax_bwd_FP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_Softmax_bwd_BFP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Full, GPU_Softmax_fwd_FP32, testing::ValuesIn(GenFullTestCasesForward()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Softmax_fwd_FP16, testing::ValuesIn(GenFullTestCasesForward()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Softmax_fwd_BFP16, testing::ValuesIn(GenFullTestCasesForward()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Softmax_bwd_FP32, testing::ValuesIn(GenFullTestCasesBackward()));
INSTANTIATE_TEST_SUITE_P(Full, GPU_Softmax_bwd_FP16, testing::ValuesIn(GenFullTestCasesBackward()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Softmax_bwd_BFP16,
                         testing::ValuesIn(GenFullTestCasesBackward()));
