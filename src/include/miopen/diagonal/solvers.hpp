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

#include "miopen/diagonal/diagflat/problem_description.hpp"
#include "miopen/execution_context.hpp"
#include <miopen/diagonal/diag/problem_description.hpp>
#include <miopen/solver.hpp>
#include <utility>

namespace miopen {

namespace solver {

namespace diagonal {

template <int N>
tensor_view_t<N - 1>
getDiagonal(const tensor_view_t<N>& tv, int64_t offset, int64_t dim1, int64_t dim2);
extern template tensor_view_t<1>
getDiagonal(const tensor_view_t<2>& tv, int64_t offset, int64_t dim1, int64_t dim2);

namespace diag {

using DiagFwdSolver =
    NonTunableSolverBase<ExecutionContext, miopen::diagonal::diag::FwdProblemDescription>;

struct DiagForward final : DiagFwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<DiagForward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::diagonal::diag::FwdProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::diagonal::diag::FwdProblemDescription& problem) const override;
};

using DiagBwdSolver =
    NonTunableSolverBase<ExecutionContext, miopen::diagonal::diag::BwdProblemDescription>;

struct DiagBackward final : DiagBwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<DiagBackward>(); }

    bool IsApplicable(const ExecutionContext& context,
                      const miopen::diagonal::diag::BwdProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::diagonal::diag::BwdProblemDescription& problem) const override;
};

} // namespace diag

namespace diagflat {

using DiagFlatFwdSolver =
    NonTunableSolverBase<ExecutionContext, miopen::diagonal::diagflat::FwdProblemDescription>;

struct DiagFlatForward final : DiagFlatFwdSolver
{
    const std::string& SolverDbId() const override { return GetSolverDbId<DiagFlatForward>(); }

    bool
    IsApplicable(const ExecutionContext& context,
                 const miopen::diagonal::diagflat::FwdProblemDescription& problem) const override;
    ConvSolution
    GetSolution(const ExecutionContext& context,
                const miopen::diagonal::diagflat::FwdProblemDescription& problem) const override;
};

} // namespace diagflat

} // namespace diagonal

} // namespace solver

} // namespace miopen
