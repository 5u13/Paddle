// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/reduce_ops/reduce_min_max_op.h"

REGISTER_REDUCE_OP(reduce_amax);
REGISTER_OP_CPU_KERNEL(
    reduce_amax,
    ops::ReduceKernel<phi::CPUContext, float, ops::MaxFunctor>,
    ops::ReduceKernel<phi::CPUContext, double, ops::MaxFunctor>,
    ops::ReduceKernel<phi::CPUContext, int, ops::MaxFunctor>,
    ops::ReduceKernel<phi::CPUContext, int64_t, ops::MaxFunctor>);
REGISTER_OP_CPU_KERNEL(
    reduce_amax_grad,
    ops::ReduceGradKernel<phi::CPUContext, float, ops::AMaxOrAMinGradFunctor>,
    ops::ReduceGradKernel<phi::CPUContext, double, ops::AMaxOrAMinGradFunctor>,
    ops::ReduceGradKernel<phi::CPUContext, int, ops::AMaxOrAMinGradFunctor>,
    ops::
        ReduceGradKernel<phi::CPUContext, int64_t, ops::AMaxOrAMinGradFunctor>);
