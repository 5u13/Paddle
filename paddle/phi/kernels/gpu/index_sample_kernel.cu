// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_sample_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

namespace {
#define PREDEFINED_BLOCK_SIZE_X 512
#define PREDEFINED_BLOCK_SIZE 1024
#define MIN(a, b) ((a) < (b) ? (a) : (b))
}  // namespace

// SUB:REF:TODO 参考它是怎么二维起kernel的
template <typename T, typename IndexT = int>
__global__ void IndexSampleForward(const IndexT* index,
                                   const T* in_data,
                                   T* out_data,
                                   size_t index_length,
                                   size_t input_length,
                                   size_t batch_size) {
  unsigned int index_i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int index_j = blockDim.y * blockIdx.y + threadIdx.y;
  for (; index_j < batch_size; index_j += blockDim.y * gridDim.y) {
    index_i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; index_i < index_length; index_i += blockDim.x * gridDim.x) {
      unsigned int index_idx = index_j * index_length + index_i;
      unsigned int in_idx = index_j * input_length + index_i;
      IndexT sample_idx = index[index_idx];
      out_data[index_idx] = in_data[in_idx - index_i + sample_idx];
    }
  }
}

// SUB:REF:TODO 参考它是怎么二维起kernel的
template <typename T, typename Context>
void IndexSampleKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& index,
                       DenseTensor* out) {
  auto index_type = index.dtype();
  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      errors::InvalidArgument(
          "Input(Index) holds the wrong type, it holds %s, but "
          "desires to be %s or %s",
          paddle::framework::DataTypeToString(
              paddle::framework::TransToProtoVarType(index_type)),
          paddle::framework::DataTypeToString(
              paddle::framework::TransToProtoVarType(DataType::INT32)),
          paddle::framework::DataTypeToString(
              paddle::framework::TransToProtoVarType((DataType::INT64)))));
  const T* in_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);
  // 这是干嘛的？GPU单独一个stream，可是还有什么stream吗
  auto stream = reinterpret_cast<const phi::GPUContext&>(ctx).stream();
  auto input_dim = x.dims();
  auto index_dim = index.dims();
  size_t batch_size = input_dim[0];
  size_t input_length = input_dim[1];
  size_t index_length = index_dim[1];

  auto block_width = paddle::platform::RoundToPowerOfTwo(index_length);
  // 二维配置时，单维度最高不超过512
  // blocksize不宜过大？通常而言，越大越好，这样单个block执行的计算量大，减少创建block的开销，同时其实也不会减损occupancy，因为本质上是shm和reg在bound occupancy，blocksize增加其实也是同步增加occupancy的，不过各代gpu最大其实就2048
  block_width = MIN(block_width, PREDEFINED_BLOCK_SIZE_X);
  int block_height =
      paddle::platform::RoundToPowerOfTwo(index_length * batch_size) /
      block_width;
  // 总之就是把blocksize限制到1024以内，这个确实是对的，因为每个sm至少要起到2个block，而且软件block最多就起1024个
  block_height = MIN(block_height, PREDEFINED_BLOCK_SIZE / block_width);
  dim3 block_dim(block_width, block_height);
  dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                (batch_size + block_dim.y - 1) / block_dim.y);
  // 这是干嘛的？就是不超过CUDA最高的gridsize
  paddle::platform::LimitGridDim(ctx, &grid_dim);

  if (index_type == DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    IndexSampleForward<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
        index_data, in_data, out_data, index_length, input_length, batch_size);
  } else if (index_type == DataType::INT32) {
    const int* index_data = index.data<int>();
    // 第三个参数shm size，第四个参数stream
    // 里面的写法也就y维度batch_size，x维度index_length
    // 二维的写法到底好在哪？二维block的局部性？相当于是blocking
    IndexSampleForward<T, int><<<grid_dim, block_dim, 0, stream>>>(
        index_data, in_data, out_data, index_length, input_length, batch_size);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_sample,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexSampleKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
