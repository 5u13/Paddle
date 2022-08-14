/* Copyright (c) 2022 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <vector>

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/fast_divmod.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/pooling.h"

namespace phi {
namespace funcs {

struct FastDivModForPooling {
 public:
  paddle::platform::FastDivMod channel;
  paddle::platform::FastDivMod width;
  paddle::platform::FastDivMod height;

  explicit HOSTDEVICE FastDivModForPooling(const int channels,
                                           const int output_width,
                                           const int output_height) {
    channel = paddle::platform::FastDivMod(channels);
    width = paddle::platform::FastDivMod(output_width);
    height = paddle::platform::FastDivMod(output_height);
  }
};

struct FastDivModForPooling3D {
public:
 paddle::platform::FastDivMod channel;
 paddle::platform::FastDivMod width;
 paddle::platform::FastDivMod height;
 paddle::platform::FastDivMod depth;

 explicit HOSTDEVICE FastDivModForPooling3D(const int channels,
                                            const int output_width,
                                            const int output_height,
                                            const int output_depth) {
   channel = paddle::platform::FastDivMod(channels);
   width = paddle::platform::FastDivMod(output_width);
   height = paddle::platform::FastDivMod(output_height);
   depth = paddle::platform::FastDivMod(output_depth);
 }
};

struct FastDivModForPooling3DStride {
 public:
  paddle::platform::FastDivMod width;
  paddle::platform::FastDivMod height;
  paddle::platform::FastDivMod depth;

  explicit HOSTDEVICE FastDivModForPooling3DStride(const int stride_width,
                                                   const int stride_height,
                                                   const int stride_depth) {
    width = paddle::platform::FastDivMod(stride_width);
    height = paddle::platform::FastDivMod(stride_height);
    depth = paddle::platform::FastDivMod(stride_depth);                      
  }
};

struct FastDivModForPoolingWithMoreStaff {
 public:
  paddle::platform::FastDivMod channel;
  paddle::platform::FastDivMod width;
  paddle::platform::FastDivMod height;
  paddle::platform::FastDivMod ksize_w;
  paddle::platform::FastDivMod ksize_h;
  paddle::platform::FastDivMod stride_w;
  paddle::platform::FastDivMod stride_h;

  explicit HOSTDEVICE FastDivModForPoolingWithMoreStaff(
      const int channels,
      const int input_width,
      const int input_height,
      const int ksize_width,
      const int ksize_height,
      const int stride_width,
      const int stride_height) {
    channel = paddle::platform::FastDivMod(channels);
    width = paddle::platform::FastDivMod(input_width);
    height = paddle::platform::FastDivMod(input_height);
    ksize_w = paddle::platform::FastDivMod(ksize_width);
    ksize_h = paddle::platform::FastDivMod(ksize_height);
    stride_w = paddle::platform::FastDivMod(stride_width);
    stride_h = paddle::platform::FastDivMod(stride_height);
  }
};

struct FastDivModForPooling3DWithMoreStaff {
public:
 paddle::platform::FastDivMod channel;
 paddle::platform::FastDivMod width;
 paddle::platform::FastDivMod height;
 paddle::platform::FastDivMod depth;
 paddle::platform::FastDivMod ksize_w;
 paddle::platform::FastDivMod ksize_h;
 paddle::platform::FastDivMod ksize_d;
 paddle::platform::FastDivMod stride_w;
 paddle::platform::FastDivMod stride_h;
 paddle::platform::FastDivMod stride_d;

 explicit HOSTDEVICE FastDivModForPooling3DWithMoreStaff(
     const int channels,
     const int input_width,
     const int input_height,
     const int input_depth,
     const int ksize_width,
     const int ksize_height,
     const int ksize_depth,
     const int stride_width,
     const int stride_height,
     const int stride_depth
    ) {
   channel = paddle::platform::FastDivMod(channels);
   width = paddle::platform::FastDivMod(input_width);
   height = paddle::platform::FastDivMod(input_height);
   depth = paddle::platform::FastDivMod(input_depth);
   ksize_w = paddle::platform::FastDivMod(ksize_width);
   ksize_h = paddle::platform::FastDivMod(ksize_height);
   ksize_d = paddle::platform::FastDivMod(ksize_depth);
   stride_w = paddle::platform::FastDivMod(stride_width);
   stride_h = paddle::platform::FastDivMod(stride_height);
   stride_d = paddle::platform::FastDivMod(stride_depth);
 }
};

// 不过如果只优化我锚定的with_index kernel，就不需要写channel_last格式的索引，但KernelPool3D&KernelPool3DGrad&KernelMaxPool3DGrad是需要的，withindex的不管是2d还是3d其实都只支持一种格式
// 不过按照2d的标准是所有的kernel都有改写的，所以还是应该写一个通用的，写优化index的kernel，没问题以后再实现到所有3d kernel
template <typename FastDivModForPooling>
__device__ void OffsetPreparationFor4Dimension(int index,
                                               bool channel_last,
                                               FastDivModForPooling divmods,
                                               const int pad_width,
                                               const int pad_height,
                                               const int aux_width,
                                               const int aux_height,
                                               int* w_offset,
                                               int* h_offset,
                                               int* c_offset,
                                               int* stride) {
  if (!channel_last) { /* NCHW */
    auto input_width_divmod = divmods.width.Divmod(index);
    auto input_height_divmod = divmods.height.Divmod(input_width_divmod.val[0]);
    auto channel_divmod = divmods.channel.Divmod(input_height_divmod.val[0]);
    *w_offset = input_width_divmod.val[1] + pad_width;
    *h_offset = input_height_divmod.val[1] + pad_height;
    *c_offset = channel_divmod.val[1];
    *stride = (channel_divmod.val[0] * divmods.channel.divisor + *c_offset) *
              aux_height * aux_width;
  } else { /* NHWC */
    auto c_divmod = divmods.channel.Divmod(index);
    auto input_width_divmod = divmods.width.Divmod(c_divmod.val[0]);
    auto input_height_divmod = divmods.height.Divmod(input_width_divmod.val[0]);
    *c_offset = c_divmod.val[1];
    *w_offset = input_width_divmod.val[1] + pad_width;
    *h_offset = input_height_divmod.val[1] + pad_height;
    *stride = input_height_divmod.val[0] * aux_height * aux_width *
              divmods.channel.divisor;
  }
}

template <typename FastDivModForPooling3D>
__device__ void OffsetPreparationFor5Dimension(int index,
                                               bool channel_last,
                                               FastDivModForPooling3D divmods,
                                               const int pad_width,
                                               const int pad_height,
                                               const int pad_depth,
                                               const int aux_width,
                                               const int aux_height,
                                               const int aux_depth,
                                               int* w_offset,
                                               int* h_offset,
                                               int* d_offset,
                                               int* c_offset,
                                               int* stride) {
    if (!channel_last) { /* NCDHW */
      auto input_width_divmod = divmods.width.Divmod(index);
      auto input_height_divmod = divmods.height.Divmod(input_width_divmod.val[0]);
      auto input_depth_divmod = divmods.depth.Divmod(input_height_divmod.val[0]);
      auto channel_divmod = divmods.channel.Divmod(input_depth_divmod.val[0]);
      *w_offset = input_width_divmod.val[1] + pad_width;
      *h_offset = input_height_divmod.val[1] + pad_height;
      *d_offset = input_depth_divmod.val[1] + pad_depth;
      *c_offset = channel_divmod.val[1];
      *stride = (channel_divmod.val[0] * divmods.channel.divisor + *c_offset) * aux_depth * aux_height * aux_width;
    } else { /* NDHWC */
      auto channel_divmod = divmods.channel.Divmod(index);
      auto input_width_divmod = divmods.width.Divmod(channel_divmod.val[0]);
      auto input_height_divmod = divmods.height.Divmod(input_width_divmod.val[0]);
      auto input_depth_divmod = divmods.depth.Divmod(input_height_divmod.val[0]);
      *c_offset = channel_divmod.val[1];
      *w_offset = input_width_divmod.val[1] + pad_width;
      *h_offset = input_height_divmod.val[1] + pad_height;
      *d_offset = input_depth_divmod.val[1] + pad_depth;
      *stride = input_depth_divmod.val[0] * aux_depth * aux_height * aux_width * divmods.channel.divisor;
    }
}

template <typename PoolProcess, typename T>
__global__ void KernelPool2D(const int nthreads,
                             const T* input_data,
                             const int channels,
                             const int input_height,
                             const int input_width,
                             const int output_height,
                             const int output_width,
                             const int ksize_height,
                             const int ksize_width,
                             const int stride_height,
                             const int stride_width,
                             const int padding_height,
                             const int padding_width,
                             FastDivModForPooling divmods,
                             PoolProcess pool_process,
                             bool exclusive,
                             bool adaptive,
                             T* output_data,
                             bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int hstart, hend, wstart, wend;
    int w_offset, h_offset, c_offset, input_offset;
    OffsetPreparationFor4Dimension<FastDivModForPooling>(index,
                                                         channel_last,
                                                         divmods,
                                                         0,
                                                         0,
                                                         input_width,
                                                         input_height,
                                                         &w_offset,
                                                         &h_offset,
                                                         &c_offset,
                                                         &input_offset);
    input_data += input_offset;

    if (adaptive) {
      hstart = AdaptStartIndex(h_offset, input_height, output_height);
      hend = AdaptEndIndex(h_offset, input_height, output_height);
      wstart = AdaptStartIndex(w_offset, input_width, output_width);
      wend = AdaptEndIndex(w_offset, input_width, output_width);
    } else {
      hstart = h_offset * stride_height - padding_height;
      hend = min(hstart + ksize_height, input_height);
      hstart = max(hstart, 0);
      wstart = w_offset * stride_width - padding_width;
      wend = min(wstart + ksize_width, input_width);
      wstart = max(wstart, 0);
    }

    T ele = pool_process.initial();
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        auto input_idx = channel_last
                             ? (h * input_width + w) * channels + c_offset
                             : h * input_width + w;
        pool_process.compute(input_data[input_idx], &ele);
      }
    }
    int pool_size = (exclusive || adaptive) ? (hend - hstart) * (wend - wstart)
                                            : ksize_height * ksize_width;
    pool_process.finalize(static_cast<T>(pool_size), &ele);
    output_data[index] = ele;
  }
}

template <typename T, typename PoolProcess>
__global__ void KernelPool2DGrad(const int nthreads,
                                 const T* __restrict__ input_data,
                                 const T* __restrict__ output_data,
                                 const T* __restrict__ output_grad,
                                 const int output_width,
                                 const int output_height,
                                 const int input_width,
                                 const int input_height,
                                 const int ksize_width,
                                 const int ksize_height,
                                 const int stride_width,
                                 const int stride_height,
                                 const int padding_width,
                                 const int padding_height,
                                 FastDivModForPoolingWithMoreStaff divmods,
                                 PoolProcess pool_process,
                                 bool exclusive,
                                 bool adaptive,
                                 T* __restrict__ input_grad,
                                 bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    T input = static_cast<T>(0);
    T input_grad_data = static_cast<T>(0);
    int phstart, phend, pwstart, pwend;
    int w_offset, h_offset, c_offset, output_offset;
    // 不是max的这个是给了padding，可是前向也没给，就反向给了，这么多个kernel就它是这样
    OffsetPreparationFor4Dimension<>(index,
                                     channel_last,
                                     divmods,
                                     padding_width,
                                     padding_height,
                                     output_width,
                                     output_height,
                                     &w_offset,
                                     &h_offset,
                                     &c_offset,
                                     &output_offset);
    if (pool_process.use_x) {
      input = input_data[index];
      output_data += output_offset;
    }
    output_grad += output_offset;

    if (adaptive) {
      // 为什么这里有还要用divmods呢？其他kernel好像不用，包括前向
      auto tmp_phend = divmods.height.Divmod((h_offset + 1) * output_height);
      auto tmp_pwend = divmods.width.Divmod((w_offset + 1) * output_width);
      phstart = divmods.height.Div(h_offset * output_height);
      pwstart = divmods.width.Div(w_offset * output_width);
      // 向上取整
      phend = tmp_phend.val[1] > 0 ? tmp_phend.val[0] + 1 : tmp_phend.val[0];
      pwend = tmp_pwend.val[1] > 0 ? tmp_pwend.val[0] + 1 : tmp_pwend.val[0];

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          auto ksize_w_divmod = divmods.ksize_w.Divmod(input_width);
          auto ksize_h_divmod = divmods.ksize_h.Divmod(input_height);
          auto tmp_width = ksize_w_divmod.val[1] > 0 ? ksize_w_divmod.val[0] + 1
                                                     : ksize_w_divmod.val[0];
          auto tmp_height = ksize_h_divmod.val[1] > 0
                                ? ksize_h_divmod.val[0] + 1
                                : ksize_h_divmod.val[0];
          int pool_size = tmp_height * tmp_width;
          int tmp_idx = ph * output_width + pw;
          int output_sub_idx =
              channel_last ? tmp_idx * divmods.channel.divisor + c_offset
                           : tmp_idx;
          T ouput_value = pool_process.use_x ? output_data[output_sub_idx]
                                             : static_cast<T>(0);
          pool_process.compute(input,
                               ouput_value,
                               output_grad[output_sub_idx],
                               static_cast<T>(1.0 / pool_size),
                               &input_grad_data);
        }
      }
    } else {
      auto stride_height_div = divmods.stride_h.Div(h_offset - ksize_height);
      auto stride_width_div = divmods.stride_w.Div(w_offset - ksize_width);
      phstart = (h_offset < ksize_height) ? 0 : stride_height_div + 1;
      pwstart = (w_offset < ksize_width) ? 0 : stride_width_div + 1;
      phend = min(divmods.stride_h.Div(h_offset) + 1, output_height);
      pwend = min(divmods.stride_w.Div(w_offset) + 1, output_width);

      if (exclusive) {
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            int hstart = ph * stride_height - padding_height;
            int wstart = pw * stride_width - padding_width;
            int hend = min(hstart + ksize_height, input_height);
            int wend = min(wstart + ksize_width, input_width);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            int pool_size = (hend - hstart) * (wend - wstart);
            int tmp_idx = ph * output_width + pw;
            int output_sub_idx =
                channel_last ? tmp_idx * divmods.channel.divisor + c_offset
                             : tmp_idx;
            T ouput_value = pool_process.use_x ? output_data[output_sub_idx]
                                               : static_cast<T>(0);
            pool_process.compute(input,
                                 ouput_value,
                                 output_grad[output_sub_idx],
                                 static_cast<T>(1.0 / pool_size),
                                 &input_grad_data);
          }
        }
      } else {
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
            int pool_size = ksize_height * ksize_width;
            int tmp_idx = ph * output_width + pw;
            int output_sub_idx =
                channel_last ? tmp_idx * divmods.channel.divisor + c_offset
                             : tmp_idx;
            T ouput_value = pool_process.use_x ? output_data[output_sub_idx]
                                               : static_cast<T>(0);
            pool_process.compute(input,
                                 ouput_value,
                                 output_grad[output_sub_idx],
                                 static_cast<T>(1.0 / pool_size),
                                 &input_grad_data);
          }
        }
      }
    }
    input_grad[index] = input_grad_data;
  }
}

template <typename T>
__global__ void KernelMaxPool2DGrad(const int nthreads,
                                    const T* input_data,
                                    const T* output_data,
                                    const T* output_grad,
                                    const int channels,
                                    const int input_height,
                                    const int input_width,
                                    const int output_height,
                                    const int output_width,
                                    const int ksize_height,
                                    const int ksize_width,
                                    const int stride_height,
                                    const int stride_width,
                                    const int padding_height,
                                    const int padding_width,
                                    T* input_grad,
                                    FastDivModForPooling divmods,
                                    bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int w_offset, h_offset, c_offset, input_offset;
    OffsetPreparationFor4Dimension<FastDivModForPooling>(index,
                                                         channel_last,
                                                         divmods,
                                                         0,
                                                         0,
                                                         input_width,
                                                         input_height,
                                                         &w_offset,
                                                         &h_offset,
                                                         &c_offset,
                                                         &input_offset);
    input_data += input_offset;
    input_grad += input_offset;

    int hstart = h_offset * stride_height - padding_height;
    int hend = min(hstart + ksize_height, input_height);
    hstart = max(hstart, 0);

    int wstart = w_offset * stride_width - padding_width;
    int wend = min(wstart + ksize_width, input_width);
    wstart = max(wstart, 0);

    T ele = output_data[index];
    int maxIndex = -1;
    bool stop = false;
    for (int h = hstart; h < hend && !stop; ++h) {
      for (int w = wstart; w < wend && !stop; ++w) {
        int input_data_idx = channel_last
                                 ? (h * input_width + w) * channels + c_offset
                                 : h * input_width + w;
        if (ele == input_data[input_data_idx]) {
          maxIndex = input_data_idx;
          stop = true;
        }
      }
    }

    if (maxIndex != -1) {
      // atomic add
      paddle::platform::CudaAtomicAdd(input_grad + maxIndex,
                                      output_grad[index]);
    }
  }
}

template <typename PoolProcess, typename T>
void Pool2dDirectCUDAFunctor<PoolProcess, T>::operator()(
    const T* input,
    const std::vector<int>& input_shape,
    const std::vector<int>& output_shape,
    const std::vector<int>& ksize,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    bool exclusive,
    bool adaptive,
    T* output,
    gpuStream_t stream,
    PoolProcess pool_compute) {
  const int batch_size = input_shape[0];
  const int input_channels = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  const int output_channels = output_shape[1];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];
  const int ksize_height = ksize[0];
  const int ksize_width = ksize[1];
  const int stride_height = strides[0];
  const int stride_width = strides[1];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];

  int nthreads = batch_size * output_channels * output_height * output_width;
  int thread_num = 1024;
#ifdef WITH_NV_JETSON
  // backends::gpu::ChangeThreadNum(context, &thread_num);
  thread_num = 512;
#endif
  int blocks = (nthreads + thread_num - 1) / thread_num;
  dim3 threads(thread_num, 1);
  dim3 grid(blocks, 1);

  auto pool_divmods =
      FastDivModForPooling(input_channels, output_width, output_height);
  KernelPool2D<PoolProcess, T><<<grid, threads, 0, stream>>>(nthreads,
                                                             input,
                                                             input_channels,
                                                             input_height,
                                                             input_width,
                                                             output_height,
                                                             output_width,
                                                             ksize_height,
                                                             ksize_width,
                                                             stride_height,
                                                             stride_width,
                                                             padding_height,
                                                             padding_width,
                                                             pool_divmods,
                                                             pool_compute,
                                                             exclusive,
                                                             adaptive,
                                                             output);
}

/*
 * Tensors are in NCHW or NHWC format.
 * Ksize, strides are two elements. These two elements represent height
 * and width, respectively.
 * Paddings are four elements. These four elements represent height_up,
 * height_down, width_left and width_right, respectively.
 */
template <typename PoolProcess, typename T>
class Pool2dFunctor<phi::GPUContext, PoolProcess, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* output,
                  PoolProcess pool_process) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    T* output_data = context.template Alloc<T>(output);

    int nthreads = batch_size * output_channels * output_height * output_width;
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    backends::gpu::ChangeThreadNum(context, &thread_num);
#endif
    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    auto pool_divmods =
        FastDivModForPooling(input_channels, output_width, output_height);
    KernelPool2D<PoolProcess, T>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 input_data,
                                                 input_channels,
                                                 input_height,
                                                 input_width,
                                                 output_height,
                                                 output_width,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_height,
                                                 stride_width,
                                                 padding_height,
                                                 padding_width,
                                                 pool_divmods,
                                                 pool_process,
                                                 exclusive,
                                                 adaptive,
                                                 output_data);
  }
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* output,
                  PoolProcess pool_process) {
    bool channel_last = (data_format == "NHWC");
    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[3] : input.dims()[1];
    const int input_height = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_width = channel_last ? input.dims()[2] : input.dims()[3];

    const int output_channels =
        channel_last ? output->dims()[3] : output->dims()[1];
    const int output_height =
        channel_last ? output->dims()[1] : output->dims()[2];
    const int output_width =
        channel_last ? output->dims()[2] : output->dims()[3];

    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];

    const int stride_height = strides[0];
    const int stride_width = strides[1];

    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    T* output_data = context.template Alloc<T>(output);

    int nthreads = batch_size * output_channels * output_height * output_width;
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    backends::gpu::ChangeThreadNum(context, &thread_num);
#endif
    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    auto pool_divmods =
        FastDivModForPooling(input_channels, output_width, output_height);
    KernelPool2D<PoolProcess, T>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 input_data,
                                                 input_channels,
                                                 input_height,
                                                 input_width,
                                                 output_height,
                                                 output_width,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_height,
                                                 stride_width,
                                                 padding_height,
                                                 padding_width,
                                                 pool_divmods,
                                                 pool_process,
                                                 exclusive,
                                                 adaptive,
                                                 output_data,
                                                 channel_last);
  }
};
/*
 * Tensors are in NCHW or NHWC format.
 * Ksize, strides are two elements. These two elements represent height
 * and width, respectively.
 * Paddings are four elements. These four elements represent height_up,
 * height_down, width_left and width_right, respectively.
 */
template <typename PoolProcess, typename T>
class Pool2dGradFunctor<phi::GPUContext, PoolProcess, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* input_grad,
                  PoolProcess pool_process) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    int nthreads = batch_size * input_channels * input_height * input_width;
    auto pool_divmods = FastDivModForPoolingWithMoreStaff(input_channels,
                                                          input_width,
                                                          input_height,
                                                          ksize_width,
                                                          ksize_height,
                                                          stride_width,
                                                          stride_height);
    
    // SUB:REF:DOING 一维grid和block的起法，只在2d的特定2个kernel使用，可以考虑用到各个kernel
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(context, nthreads);
    KernelPool2DGrad<T, PoolProcess><<<config.block_per_grid,
                                       config.thread_per_block,
                                       0,
                                       context.stream()>>>(nthreads,
                                                           input_data,
                                                           output_data,
                                                           output_grad_data,
                                                           output_width,
                                                           output_height,
                                                           input_width,
                                                           input_height,
                                                           ksize_width,
                                                           ksize_height,
                                                           stride_width,
                                                           stride_height,
                                                           padding_width,
                                                           padding_height,
                                                           pool_divmods,
                                                           pool_process,
                                                           exclusive,
                                                           adaptive,
                                                           input_grad_data);
  }
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* input_grad,
                  PoolProcess pool_process) {
    bool channel_last = (data_format == "NHWC");

    const int batch_size = input.dims()[0];
    const int input_channels = channel_last ? input.dims()[3] : input.dims()[1];
    const int input_height = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_width = channel_last ? input.dims()[2] : input.dims()[3];

    const int output_channels =
        channel_last ? output.dims()[3] : output.dims()[1];
    const int output_height =
        channel_last ? output.dims()[1] : output.dims()[2];
    const int output_width = channel_last ? output.dims()[2] : output.dims()[3];

    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];

    const int stride_height = strides[0];
    const int stride_width = strides[1];

    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    int nthreads = batch_size * input_channels * input_height * input_width;
    auto pool_divmods = FastDivModForPoolingWithMoreStaff(input_channels,
                                                          input_width,
                                                          input_height,
                                                          ksize_width,
                                                          ksize_height,
                                                          stride_width,
                                                          stride_height);

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(context, nthreads);
    KernelPool2DGrad<T, PoolProcess><<<config.block_per_grid,
                                       config.thread_per_block,
                                       0,
                                       context.stream()>>>(nthreads,
                                                           input_data,
                                                           output_data,
                                                           output_grad_data,
                                                           output_width,
                                                           output_height,
                                                           input_width,
                                                           input_height,
                                                           ksize_width,
                                                           ksize_height,
                                                           stride_width,
                                                           stride_height,
                                                           padding_width,
                                                           padding_height,
                                                           pool_divmods,
                                                           pool_process,
                                                           exclusive,
                                                           adaptive,
                                                           input_grad_data,
                                                           channel_last);
  }
};

/*
 * Tensors are in NCHW or NHWC format.
 * Ksize, strides are two elements. These two elements represent height
 * and width, respectively.
 * Paddings are four elements. These four elements represent height_up,
 * height_down, width_left and width_right, respectively.
 */
template <typename T>
class MaxPool2dGradFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  DenseTensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    auto pool_divmods =
        FastDivModForPooling(input_channels, output_width, output_height);
    KernelMaxPool2DGrad<T>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 input_data,
                                                 output_data,
                                                 output_grad_data,
                                                 input_channels,
                                                 input_height,
                                                 input_width,
                                                 output_height,
                                                 output_width,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_height,
                                                 stride_width,
                                                 padding_height,
                                                 padding_width,
                                                 input_grad_data,
                                                 pool_divmods);
  }
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  DenseTensor* input_grad) {
    bool channel_last = (data_format == "NHWC");

    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[3] : input.dims()[1];
    const int input_height = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_width = channel_last ? input.dims()[2] : input.dims()[3];

    const int output_channels =
        channel_last ? output.dims()[3] : output.dims()[1];
    const int output_height =
        channel_last ? output.dims()[1] : output.dims()[2];
    const int output_width = channel_last ? output.dims()[2] : output.dims()[3];

    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];

    const int stride_height = strides[0];
    const int stride_width = strides[1];

    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    auto pool_divmods =
        FastDivModForPooling(input_channels, output_width, output_height);

    KernelMaxPool2DGrad<T>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 input_data,
                                                 output_data,
                                                 output_grad_data,
                                                 input_channels,
                                                 input_height,
                                                 input_width,
                                                 output_height,
                                                 output_width,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_height,
                                                 stride_width,
                                                 padding_height,
                                                 padding_width,
                                                 input_grad_data,
                                                 pool_divmods,
                                                 channel_last);
  }
};

template class Pool2dDirectCUDAFunctor<MaxPool<float>, float>;
template class Pool2dDirectCUDAFunctor<AvgPool<float>, float>;

template class MaxPool2dGradFunctor<phi::GPUContext, float>;
template class MaxPool2dGradFunctor<phi::GPUContext, double>;
template class MaxPool2dGradFunctor<phi::GPUContext, dtype::float16>;

template class Pool2dFunctor<phi::GPUContext, MaxPool<float>, float>;
template class Pool2dFunctor<phi::GPUContext, AvgPool<float>, float>;
template class Pool2dGradFunctor<phi::GPUContext, MaxPoolGrad<float>, float>;
template class Pool2dGradFunctor<phi::GPUContext, AvgPoolGrad<float>, float>;
template class Pool2dFunctor<phi::GPUContext, MaxPool<double>, double>;
template class Pool2dFunctor<phi::GPUContext, AvgPool<double>, double>;
template class Pool2dGradFunctor<phi::GPUContext, MaxPoolGrad<double>, double>;
template class Pool2dGradFunctor<phi::GPUContext, AvgPoolGrad<double>, double>;

template class Pool2dFunctor<phi::GPUContext,
                             MaxPool<dtype::float16>,
                             dtype::float16>;
template class Pool2dFunctor<phi::GPUContext,
                             AvgPool<dtype::float16>,
                             dtype::float16>;
template class Pool2dGradFunctor<phi::GPUContext,
                                 MaxPoolGrad<dtype::float16>,
                                 dtype::float16>;
template class Pool2dGradFunctor<phi::GPUContext,
                                 AvgPoolGrad<dtype::float16>,
                                 dtype::float16>;

template <typename PoolProcess, typename T>
__global__ void KernelPool3D(const int nthreads,
                             const T* input_data,
                             const int channels,
                             const int input_depth,
                             const int input_height,
                             const int input_width,
                             const int output_depth,
                             const int output_height,
                             const int output_width,
                             const int ksize_depth,
                             const int ksize_height,
                             const int ksize_width,
                             const int stride_depth,
                             const int stride_height,
                             const int stride_width,
                             const int padding_depth,
                             const int padding_height,
                             const int padding_width,
                             PoolProcess pool_process,
                             bool exclusive,
                             bool adaptive,
                             T* output_data,
                             bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int pw, ph, pd, c, batch_idx;
    // 这个才是合乎我的逻辑的index方式，直接根据output来计算
    if (!channel_last) {
      pw = index % output_width;
      ph = (index / output_width) % output_height;
      pd = (index / output_width / output_height) % output_depth;
      c = (index / output_width / output_height / output_depth) % channels;
      batch_idx =
          index / output_width / output_height / output_depth / channels;
    } else {
      c = index % channels;
      pw = (index / channels) % output_width;
      ph = (index / channels / output_width) % output_height;
      pd = (index / channels / output_width / output_height) % output_depth;
      batch_idx =
          index / channels / output_width / output_height / output_depth;
    }

    int dstart, dend;
    int hstart, hend;
    int wstart, wend;
    if (adaptive) {
      dstart = AdaptStartIndex(pd, input_depth, output_depth);
      dend = AdaptEndIndex(pd, input_depth, output_depth);

      hstart = AdaptStartIndex(ph, input_height, output_height);
      hend = AdaptEndIndex(ph, input_height, output_height);

      wstart = AdaptStartIndex(pw, input_width, output_width);
      wend = AdaptEndIndex(pw, input_width, output_width);
    } else {
      dstart = pd * stride_depth - padding_depth;
      hstart = ph * stride_height - padding_height;
      wstart = pw * stride_width - padding_width;
      dend = min(dstart + ksize_depth, input_depth);
      hend = min(hstart + ksize_height, input_height);
      wend = min(wstart + ksize_width, input_width);
      dstart = max(dstart, 0);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
    }

    int input_data_stride;
    if (!channel_last) { /* NCDHW */
      input_data_stride =
          (batch_idx * channels + c) * input_depth * input_height * input_width;
    } else { /* NDHWC */
      input_data_stride =
          batch_idx * input_depth * input_height * input_width * channels;
    }
    input_data += input_data_stride;

    // 就是根据pool的类型（max，avg）创建的一个模版类，用于初始化元素值，对元素值求max/pool
    T ele = pool_process.initial();
    // 每个index对应output一个元素，output一个元素对应在input网格上作用一个kernel
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          auto input_data_idx =
              channel_last
                  ? ((d * input_height + h) * input_width + w) * channels + c
                  : (d * input_height + h) * input_width + w;
          pool_process.compute(input_data[input_data_idx], &ele);
        }
      }
    }
    int pool_size = (exclusive || adaptive)
                        ? (dend - dstart) * (hend - hstart) * (wend - wstart)
                        : ksize_depth * ksize_height * ksize_width;
    pool_process.finalize(static_cast<T>(pool_size), &ele);
    output_data[index] = ele;
  }
}

template <typename T, typename PoolProcess>
__global__ void KernelPool3DGrad(const int nthreads,
                                 const T* __restrict__ input_data,
                                 const T* __restrict__ output_data,
                                 const T* __restrict__ output_grad,
                                 const int channels,
                                 const int input_depth,
                                 const int input_height,
                                 const int input_width,
                                 const int output_depth,
                                 const int output_height,
                                 const int output_width,
                                 const int ksize_depth,
                                 const int ksize_height,
                                 const int ksize_width,
                                 const int stride_depth,
                                 const int stride_height,
                                 const int stride_width,
                                 const int padding_depth,
                                 const int padding_height,
                                 const int padding_width,
                                 PoolProcess pool_process,
                                 bool exclusive,
                                 bool adaptive,
                                 T* input_grad,
                                 bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int w_offset, h_offset, d_offset, c_offset, batch_idx, output_stride;
    T input = static_cast<T>(0);
    if (!channel_last) { /* "NCDHW" */
      w_offset = index % input_width + padding_width;
      h_offset = (index / input_width) % input_height + padding_height;
      d_offset =
          (index / input_width / input_height) % input_depth + padding_depth;
      c_offset = (index / input_width / input_height / input_depth) % channels;
      batch_idx = index / input_width / input_height / input_depth / channels;
      output_stride = (batch_idx * channels + c_offset) * output_depth *
                      output_height * output_width;
    } else { /* "NDHWC" */
      c_offset = index % channels;
      w_offset = (index / channels) % input_width + padding_width;
      h_offset =
          (index / channels / input_width) % input_height + padding_height;
      d_offset = (index / channels / input_width / input_height) % input_depth +
                 padding_depth;
      batch_idx = index / channels / input_width / input_height / input_depth;
      output_stride =
          batch_idx * output_depth * output_height * output_width * channels;
    }

    int pdstart, pdend;
    int phstart, phend;
    int pwstart, pwend;
    if (adaptive) {
      pdstart = AdaptStartIndex(d_offset, output_depth, input_depth);
      pdend = AdaptEndIndex(d_offset, output_depth, input_depth);

      phstart = AdaptStartIndex(h_offset, output_height, input_height);
      phend = AdaptEndIndex(h_offset, output_height, input_height);

      pwstart = AdaptStartIndex(w_offset, output_width, input_width);
      pwend = AdaptEndIndex(w_offset, output_width, input_width);
    } else {
      pdstart = (d_offset < ksize_depth)
                    ? 0
                    : (d_offset - ksize_depth) / stride_depth + 1;
      phstart = (h_offset < ksize_height)
                    ? 0
                    : (h_offset - ksize_height) / stride_height + 1;
      pwstart = (w_offset < ksize_width)
                    ? 0
                    : (w_offset - ksize_width) / stride_width + 1;
      pdend = min((d_offset) / stride_depth + 1, output_depth);
      phend = min((h_offset) / stride_height + 1, output_height);
      pwend = min((w_offset) / stride_width + 1, output_width);
    }
    if (pool_process.use_x) {
      input = input_data[index];
      output_data += output_stride;
    }
    output_grad += output_stride;
    T input_grad_data = static_cast<T>(0.0);

    for (int pd = pdstart; pd < pdend; ++pd) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // figure out the pooling size
          int pool_size;
          if (adaptive) {
            pool_size =
                static_cast<int>(
                    ceil(static_cast<double>(input_depth) / ksize_depth)) *
                static_cast<int>(
                    ceil(static_cast<double>(input_height) / ksize_height)) *
                static_cast<int>(
                    ceil(static_cast<double>(input_width) / ksize_width));
          } else {
            int dstart = pd * stride_depth - padding_depth;
            int hstart = ph * stride_height - padding_height;
            int wstart = pw * stride_width - padding_width;
            int dend = min(dstart + ksize_depth, input_depth);
            int hend = min(hstart + ksize_height, input_height);
            int wend = min(wstart + ksize_width, input_width);
            dstart = max(dstart, 0);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            pool_size =
                exclusive ? (dend - dstart) * (hend - hstart) * (wend - wstart)
                          : ksize_depth * ksize_height * ksize_width;
          }

          int output_sub_idx =
              channel_last
                  ? ((pd * output_height + ph) * output_width + pw) * channels +
                        c_offset
                  : (pd * output_height + ph) * output_width + pw;
          T ouput_value = pool_process.use_x ? output_data[output_sub_idx]
                                             : static_cast<T>(0);
          pool_process.compute(input,
                               ouput_value,
                               output_grad[output_sub_idx],
                               static_cast<T>(1.0 / pool_size),
                               &input_grad_data);
        }
      }
    }
    input_grad[index] = input_grad_data;
  }
}

template <typename T>
__global__ void KernelMaxPool3DGrad(const int nthreads,
                                    const T* input_data,
                                    const T* output_data,
                                    const T* output_grad,
                                    const int channels,
                                    const int input_depth,
                                    const int input_height,
                                    const int input_width,
                                    const int output_depth,
                                    const int output_height,
                                    const int output_width,
                                    const int ksize_depth,
                                    const int ksize_height,
                                    const int ksize_width,
                                    const int stride_depth,
                                    const int stride_height,
                                    const int stride_width,
                                    const int padding_depth,
                                    const int padding_height,
                                    const int padding_width,
                                    T* input_grad,
                                    bool channel_last = false) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int pw, ph, pd, c, batch_idx;

    if (!channel_last) { /*NCDHW*/
      pw = index % output_width;
      ph = (index / output_width) % output_height;
      pd = (index / output_width / output_height) % output_depth;
      c = (index / output_width / output_height / output_depth) % channels;
      batch_idx =
          index / output_width / output_height / output_depth / channels;
    } else { /*NDHWC*/
      c = index % channels;
      pw = (index / channels) % output_width;
      ph = (index / channels / output_width) % output_height;
      pd = (index / channels / output_width / output_height) % output_depth;
      batch_idx =
          index / channels / output_width / output_height / output_depth;
    }

    int dstart = pd * stride_depth - padding_depth;
    int hstart = ph * stride_height - padding_height;
    int wstart = pw * stride_width - padding_width;

    int dend = min(dstart + ksize_depth, input_depth);
    int hend = min(hstart + ksize_height, input_height);
    int wend = min(wstart + ksize_width, input_width);

    dstart = max(dstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    T ele = output_data[index];
    bool stop = false;
    int maxIdx = -1;

    int input_stride;
    if (!channel_last) {
      input_stride =
          (batch_idx * channels + c) * input_depth * input_height * input_width;
    } else {
      input_stride =
          batch_idx * input_depth * input_height * input_width * channels;
    }
    input_data += input_stride;
    input_grad += input_stride;
    // SUB:REF:DOING 检查output对应的input数据，如果正好等于当前output数据，说明这个是input的maxindex
    // 为什么不跟pytorch一样直接取mask
    for (int d = dstart; d < dend && !stop; ++d) {
      for (int h = hstart; h < hend && !stop; ++h) {
        for (int w = wstart; w < wend && !stop; ++w) {
          int input_data_idx =
              channel_last
                  ? ((d * input_height + h) * input_width + w) * channels + c
                  : (d * input_height + h) * input_width + w;
          if (ele == input_data[input_data_idx]) {
            stop = true;
            maxIdx = input_data_idx;
          }
        }
      }
    }
    if (maxIdx != -1) {
      // atomic add
      paddle::platform::CudaAtomicAdd(input_grad + maxIdx, output_grad[index]);
    }
  }
}

template <typename PoolProcess, typename T>
void Pool3dDirectCUDAFunctor<PoolProcess, T>::operator()(
    const T* input,
    const std::vector<int>& input_shape,
    const std::vector<int>& output_shape,
    const std::vector<int>& ksize,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    bool exclusive,
    bool adaptive,
    T* output,
    gpuStream_t stream,
    PoolProcess pool_compute) {
  const int batch_size = input_shape[0];
  const int input_channels = input_shape[1];
  const int input_depth = input_shape[2];
  const int input_height = input_shape[3];
  const int input_width = input_shape[4];
  const int output_channels = output_shape[1];
  const int output_depth = output_shape[2];
  const int output_height = output_shape[3];
  const int output_width = output_shape[4];
  const int ksize_depth = ksize[0];
  const int ksize_height = ksize[1];
  const int ksize_width = ksize[2];
  const int stride_depth = strides[0];
  const int stride_height = strides[1];
  const int stride_width = strides[2];
  const int padding_depth = paddings[0];
  const int padding_height = paddings[1];
  const int padding_width = paddings[2];

  int nthreads = batch_size * output_channels * output_depth * output_height *
                 output_width;
  int thread_num = 1024;
#ifdef WITH_NV_JETSON
  thread_num = 512;
#endif
  int blocks = (nthreads + thread_num - 1) / thread_num;
  dim3 threads(thread_num, 1);
  dim3 grid(blocks, 1);

  KernelPool3D<PoolProcess, T><<<grid, threads, 0, stream>>>(nthreads,
                                                             input,
                                                             input_channels,
                                                             input_depth,
                                                             input_height,
                                                             input_width,
                                                             output_depth,
                                                             output_height,
                                                             output_width,
                                                             ksize_depth,
                                                             ksize_height,
                                                             ksize_width,
                                                             stride_depth,
                                                             stride_height,
                                                             stride_width,
                                                             padding_depth,
                                                             padding_height,
                                                             padding_width,
                                                             pool_compute,
                                                             exclusive,
                                                             adaptive,
                                                             output);
}

/*
 * Tensors are in NCDHW or NDHWC format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 * Paddings are six elements. These six elements represent depth_forth,
 * depth_back,
 * height_up, height_down, width_left and width_right, respectively.
 */
template <typename PoolProcess, class T>
class Pool3dFunctor<phi::GPUContext, PoolProcess, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* output,
                  PoolProcess pool_process) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output->dims()[1];
    const int output_depth = output->dims()[2];
    const int output_height = output->dims()[3];
    const int output_width = output->dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T* input_data = input.data<T>();
    T* output_data = context.template Alloc<T>(output);

    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    backends::gpu::ChangeThreadNum(context, &thread_num);
#endif
    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    KernelPool3D<PoolProcess, T>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 input_data,
                                                 input_channels,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_depth,
                                                 output_height,
                                                 output_width,
                                                 ksize_depth,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_depth,
                                                 stride_height,
                                                 stride_width,
                                                 padding_depth,
                                                 padding_height,
                                                 padding_width,
                                                 pool_process,
                                                 exclusive,
                                                 adaptive,
                                                 output_data);
  }
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* output,
                  PoolProcess pool_process) {
    bool channel_last = (data_format == "NDHWC");
    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[4] : input.dims()[1];
    const int input_depth = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_height = channel_last ? input.dims()[2] : input.dims()[3];
    const int input_width = channel_last ? input.dims()[3] : input.dims()[4];

    const int output_channels =
        channel_last ? output->dims()[4] : output->dims()[1];
    const int output_depth =
        channel_last ? output->dims()[1] : output->dims()[2];
    const int output_height =
        channel_last ? output->dims()[2] : output->dims()[3];
    const int output_width =
        channel_last ? output->dims()[3] : output->dims()[4];

    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];

    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];

    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T* input_data = input.data<T>();
    T* output_data = context.template Alloc<T>(output);

    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    backends::gpu::ChangeThreadNum(context, &thread_num);
#endif
    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    KernelPool3D<PoolProcess, T>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 input_data,
                                                 input_channels,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_depth,
                                                 output_height,
                                                 output_width,
                                                 ksize_depth,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_depth,
                                                 stride_height,
                                                 stride_width,
                                                 padding_depth,
                                                 padding_height,
                                                 padding_width,
                                                 pool_process,
                                                 exclusive,
                                                 adaptive,
                                                 output_data,
                                                 channel_last);
  }
};

/*
 * Tensors are in NCDHW or NDHWC format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 * Paddings are six elements. These six elements represent depth_forth,
 * depth_back,
 * height_up, height_down, width_left and width_right, respectively.
 */
template <typename PoolProcess, class T>
class Pool3dGradFunctor<phi::GPUContext, PoolProcess, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* input_grad,
                  PoolProcess pool_process) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output.dims()[1];
    const int output_depth = output.dims()[2];
    const int output_height = output.dims()[3];
    const int output_width = output.dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    int nthreads =
        batch_size * input_channels * input_depth * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool3DGrad<T, PoolProcess>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 input_data,
                                                 output_data,
                                                 output_grad_data,
                                                 input_channels,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_depth,
                                                 output_height,
                                                 output_width,
                                                 ksize_depth,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_depth,
                                                 stride_height,
                                                 stride_width,
                                                 padding_depth,
                                                 padding_height,
                                                 padding_width,
                                                 pool_process,
                                                 exclusive,
                                                 adaptive,
                                                 input_grad_data);
  }
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  bool exclusive,
                  bool adaptive,
                  DenseTensor* input_grad,
                  PoolProcess pool_process) {
    bool channel_last = (data_format == "NDHWC");

    const int batch_size = input.dims()[0];
    const int input_channels = channel_last ? input.dims()[4] : input.dims()[1];
    const int input_depth = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_height = channel_last ? input.dims()[2] : input.dims()[3];
    const int input_width = channel_last ? input.dims()[3] : input.dims()[4];

    const int output_channels =
        channel_last ? output.dims()[4] : output.dims()[1];
    const int output_depth = channel_last ? output.dims()[1] : output.dims()[2];
    const int output_height =
        channel_last ? output.dims()[2] : output.dims()[3];
    const int output_width = channel_last ? output.dims()[3] : output.dims()[4];

    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];

    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];

    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    int nthreads =
        batch_size * input_channels * input_depth * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool3DGrad<T, PoolProcess><<<grid, threads, 0, context.stream()>>>(
        nthreads,
        input_data,
        output_data,
        output_grad_data,
        input_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        ksize_depth,
        ksize_height,
        ksize_width,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        pool_process,
        exclusive,
        adaptive,
        input_grad_data,
        channel_last);  // add channel_last
  }
};

/*
 * tensors are in NCDHW or NDHWC format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 * Paddings are six elements. These six elements represent depth_forth,
 * depth_back,
 * height_up, height_down, width_left and width_right, respectively.
 */
template <class T>
class MaxPool3dGradFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  DenseTensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output.dims()[1];
    const int output_depth = output.dims()[2];
    const int output_height = output.dims()[3];
    const int output_width = output.dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool3DGrad<T>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 input_data,
                                                 output_data,
                                                 output_grad_data,
                                                 input_channels,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_depth,
                                                 output_height,
                                                 output_width,
                                                 ksize_depth,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_depth,
                                                 stride_height,
                                                 stride_width,
                                                 padding_depth,
                                                 padding_height,
                                                 padding_width,
                                                 input_grad_data);
  }
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format,
                  DenseTensor* input_grad) {
    bool channel_last = (data_format == "NDHWC");
    const int batch_size = input.dims()[0];

    const int input_channels = channel_last ? input.dims()[4] : input.dims()[1];
    const int input_depth = channel_last ? input.dims()[1] : input.dims()[2];
    const int input_height = channel_last ? input.dims()[2] : input.dims()[3];
    const int input_width = channel_last ? input.dims()[3] : input.dims()[4];

    const int output_channels =
        channel_last ? output.dims()[4] : output.dims()[1];
    const int output_depth = channel_last ? output.dims()[1] : output.dims()[2];
    const int output_height =
        channel_last ? output.dims()[2] : output.dims()[3];
    const int output_width = channel_last ? output.dims()[3] : output.dims()[4];

    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];

    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];

    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T* input_data = input.data<T>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = context.template Alloc<T>(input_grad);

    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelMaxPool3DGrad<T><<<grid, threads, 0, context.stream()>>>(
        nthreads,
        input_data,
        output_data,
        output_grad_data,
        input_channels,
        input_depth,
        input_height,
        input_width,
        output_depth,
        output_height,
        output_width,
        ksize_depth,
        ksize_height,
        ksize_width,
        stride_depth,
        stride_height,
        stride_width,
        padding_depth,
        padding_height,
        padding_width,
        input_grad_data,
        channel_last);  // add channel_last
  }
};

template class Pool3dDirectCUDAFunctor<MaxPool<float>, float>;
template class Pool3dDirectCUDAFunctor<AvgPool<float>, float>;

template class MaxPool3dGradFunctor<phi::GPUContext, float>;
template class MaxPool3dGradFunctor<phi::GPUContext, double>;
template class MaxPool3dGradFunctor<phi::GPUContext, dtype::float16>;

template class Pool3dFunctor<phi::GPUContext, MaxPool<float>, float>;
template class Pool3dFunctor<phi::GPUContext, AvgPool<float>, float>;
template class Pool3dGradFunctor<phi::GPUContext, MaxPoolGrad<float>, float>;
template class Pool3dGradFunctor<phi::GPUContext, AvgPoolGrad<float>, float>;
template class Pool3dFunctor<phi::GPUContext, MaxPool<double>, double>;
template class Pool3dFunctor<phi::GPUContext, AvgPool<double>, double>;
template class Pool3dGradFunctor<phi::GPUContext, MaxPoolGrad<double>, double>;
template class Pool3dGradFunctor<phi::GPUContext, AvgPoolGrad<double>, double>;

template class Pool3dFunctor<phi::GPUContext,
                             MaxPool<dtype::float16>,
                             dtype::float16>;
template class Pool3dFunctor<phi::GPUContext,
                             AvgPool<dtype::float16>,
                             dtype::float16>;
template class Pool3dGradFunctor<phi::GPUContext,
                                 MaxPoolGrad<dtype::float16>,
                                 dtype::float16>;
template class Pool3dGradFunctor<phi::GPUContext,
                                 AvgPoolGrad<dtype::float16>,
                                 dtype::float16>;

// SUB:REF:DOING maxpool2d前向kernel
template <typename T1, typename T2>
__global__ void KernelMaxPool2dWithIdx(const int nthreads,
                                       const T1* input_data,
                                       const int channels,
                                       const int input_height,
                                       const int input_width,
                                       const int output_height,
                                       const int output_width,
                                       const int ksize_height,
                                       const int ksize_width,
                                       const int stride_height,
                                       const int stride_width,
                                       const int padding_height,
                                       const int padding_width,
                                       bool adaptive,
                                       T1* output_data,
                                       T2* mask_data,
                                       FastDivModForPooling divmods) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int hstart, hend, wstart, wend;
    int w_offset, h_offset, c_offset, input_offset;
    // 第二个参数是channel_last，第四个和第五个参数是pad_width和pad_height
    // __device__ void OffsetPreparationFor4Dimension(int index, bool channel_last, FastDivModForPooling divmods, const int pad_width, const int pad_height, const int aux_width, const int aux_height, int* w_offset, int* h_offset, int* c_offset, int* stride)
    // 这里pad给0有点奇怪，不过其实应该也取决于后面怎么算，这里后面又去减pad了
    OffsetPreparationFor4Dimension<FastDivModForPooling>(index,
                                                         false,
                                                         divmods,
                                                         0,
                                                         0,
                                                         input_width,
                                                         input_height,
                                                         &w_offset,
                                                         &h_offset,
                                                         &c_offset,
                                                         &input_offset);
    // SUB:FIXME 这么写不对吧，如果多次循环的话，index每次更新以后算出来的并不是stride，而是相对传参进来的位置的offset
    // 原本写的kernel的一维线程配置刚好也是计算好只循环1次的
    input_data += input_offset;

    // 这里前向是用了adaptive的分装好的index计算接口
    if (adaptive) {
      hstart = AdaptStartIndex(h_offset, input_height, output_height);
      hend = AdaptEndIndex(h_offset, input_height, output_height);

      wstart = AdaptStartIndex(w_offset, input_width, output_width);
      wend = AdaptEndIndex(w_offset, input_width, output_width);
    } else {
      // offset是相对output算的，对于input，kernel作用范围要往回走padding
      hstart = h_offset * stride_height - padding_height;
      hend = min(hstart + ksize_height, input_height);
      hstart = max(hstart, 0);

      wstart = w_offset * stride_width - padding_width;
      wend = min(wstart + ksize_width, input_width);
      wstart = max(wstart, 0);
    }

    T1 ele = -FLT_MAX;
    int max_index = -1;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_index = h * input_width + w;
        if (ele < input_data[input_index]) {
          max_index = input_index;
          ele = input_data[input_index];
        }
      }
    }
    output_data[index] = ele;
    mask_data[index] = max_index;
  }
}

// SUB:REF:DOING maxpool2d反向kernel
template <typename T1, typename T2>
__global__ void KernelMaxPool2DWithIdxGrad(const int nthreads,
                                           const T1* output_grad,
                                           const T2* mask_data,
                                           const int channels,
                                           const int input_height,
                                           const int input_width,
                                           const int output_height,
                                           const int output_width,
                                           const int ksize_height,
                                           const int ksize_width,
                                           const int stride_height,
                                           const int stride_width,
                                           const int padding_height,
                                           const int padding_width,
                                           bool adaptive,
                                           T1* input_grad,
                                           FastDivModForPooling divmods) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int phstart, phend, pwstart, pwend;
    int w_offset, h_offset, c_offset, output_offset;
    // 跟前向一样也没有给pad
    OffsetPreparationFor4Dimension<FastDivModForPooling>(index,
                                                         false,
                                                         divmods,
                                                         0,
                                                         0,
                                                         output_width,
                                                         output_height,
                                                         &w_offset,
                                                         &h_offset,
                                                         &c_offset,
                                                         &output_offset);
    // SUB:FIXME 这么写不对吧，如果多次循环的话，index每次更新以后算出来的并不是stride，而是相对传参进来的位置的offset
    mask_data += output_offset;
    output_grad += output_offset;

    // 反向没有用adaptive索引计算的封装接口
    if (adaptive) {
      phstart = h_offset * output_height / input_height;
      phend =
          min((h_offset + 1) * output_height / input_height + 1, output_height);
      pwstart = w_offset * output_width / input_width;
      pwend =
          min((w_offset + 1) * output_width / input_width + 1, output_width);
    } else {
      // 这里是要算output即grad_input的索引，所以要加padding
      // 注意output_grad指的就是output的梯度
      phstart =
          (h_offset + padding_height < ksize_height)
              ? 0
              : (h_offset + padding_height - ksize_height) / stride_height + 1;
      pwstart =
          (w_offset + padding_width < ksize_width)
              ? 0
              : (w_offset + padding_width - ksize_width) / stride_width + 1;
      phend =
          min((h_offset + padding_height) / stride_height + 1, output_height);
      pwend = min((w_offset + padding_width) / stride_width + 1, output_width);
    }

    T1 input_grad_data = 0;
    int input_current_featuremap_idx = h_offset * input_width + w_offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_data[ph * output_width + pw] == input_current_featuremap_idx)
          input_grad_data += output_grad[ph * output_width + pw];
      }
    }
    // 根据input切的grid和block
    // input的这个位置有没有梯度，其实也是取决于output上所有它可能作用到的位置是否是max，求和
    input_grad[index] = input_grad_data;
  }
}

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
 // SUB:REF:DOING maxpool2d起前向
template <typename T1, typename T2>
class MaxPool2dWithIndexFunctor<phi::GPUContext, T1, T2> {
 public:
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool adaptive,
                  DenseTensor* output,
                  DenseTensor* mask) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T1* input_data = input.data<T1>();
    T1* output_data = context.template Alloc<T1>(output);
    T2* mask_data = context.template Alloc<T2>(mask);

    int nthreads = batch_size * output_channels * output_height * output_width;
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    backends::gpu::ChangeThreadNum(context, &thread_num);
#endif

    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    // 为什么只需要这三个参数呢。。如果input_channels是output_channels的话倒还能理解
    // 不过注意看output_channels都没有传到前向kernel里，对于pooling操作，input_channels==output_channels吧，没有index的2d前向同样也是这么写的
    auto pool_divmods =
        FastDivModForPooling(input_channels, output_width, output_height);
    // 这里要把pool_divmods作为参数给进去
    KernelMaxPool2dWithIdx<T1, T2>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 input_data,
                                                 input_channels,
                                                 input_height,
                                                 input_width,
                                                 output_height,
                                                 output_width,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_height,
                                                 stride_width,
                                                 padding_height,
                                                 padding_width,
                                                 adaptive,
                                                 output_data,
                                                 mask_data,
                                                 pool_divmods);
  }
};

/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
 // SUB:REF:DOING maxpool2d起反向
template <typename T1, typename T2>
class MaxPool2dWithIndexGradFunctor<phi::GPUContext, T1, T2> {
 public:
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& output_grad,
                  const DenseTensor& mask,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool adaptive,
                  DenseTensor* input_grad) {
    const int batch_size = input_grad->dims()[0];
    const int input_channels = input_grad->dims()[1];
    const int input_height = input_grad->dims()[2];
    const int input_width = input_grad->dims()[3];
    const int output_height = output_grad.dims()[2];
    const int output_width = output_grad.dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T2* mask_data = mask.data<T2>();
    const T1* output_grad_data = output_grad.data<T1>();
    T1* input_grad_data = context.template Alloc<T1>(input_grad);

    int nthreads = batch_size * input_channels * input_height * input_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    // 跟前向逻辑是一样的，就是把计算索引需要的output传进去，对于反向而言input就是output
    auto pool_divmods =
        FastDivModForPooling(input_channels, input_width, input_height);
    KernelMaxPool2DWithIdxGrad<T1, T2>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 output_grad_data,
                                                 mask_data,
                                                 input_channels,
                                                 input_height,
                                                 input_width,
                                                 output_height,
                                                 output_width,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_height,
                                                 stride_width,
                                                 padding_height,
                                                 padding_width,
                                                 adaptive,
                                                 input_grad_data,
                                                 pool_divmods);
  }
};

template class MaxPool2dWithIndexFunctor<phi::GPUContext, float, int>;
template class MaxPool2dWithIndexGradFunctor<phi::GPUContext, float, int>;
template class MaxPool2dWithIndexFunctor<phi::GPUContext, double, int>;
template class MaxPool2dWithIndexGradFunctor<phi::GPUContext, double, int>;

// SUB:DONE maxpool3d前向kernel
/*
template <typename T1, typename T2>
__global__ void KernelMaxPool3DWithIdx(const int nthreads,
                                       const T1* input_data,
                                       const int channels,
                                       const int input_depth,
                                       const int input_height,
                                       const int input_width,
                                       const int output_depth,
                                       const int output_height,
                                       const int output_width,
                                       const int ksize_depth,
                                       const int ksize_height,
                                       const int ksize_width,
                                       const int stride_depth,
                                       const int stride_height,
                                       const int stride_width,
                                       const int padding_depth,
                                       const int padding_height,
                                       const int padding_width,
                                       bool adaptive,
                                       T1* output_data,
                                       T2* mask_data,
                                       FastDivModForPooling3D divmods) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int dstart, dend, hstart, hend, wstart, wend;
    int w_offset, h_offset, d_offset, c_offset, input_offset;
    OffsetPreparationFor5Dimension<FastDivModForPooling3D>(index, 
                                                           false, 
                                                           divmods, 
                                                           0, 
                                                           0,
                                                           0,
                                                           input_width, 
                                                           input_height, 
                                                           input_depth, 
                                                           &w_offset, 
                                                           &h_offset, 
                                                           &d_offset, 
                                                           &c_offset, 
                                                           &input_offset);
    input_data += input_offset;

    if (adaptive) {
      dstart = AdaptStartIndex(d_offset, input_depth, output_depth);
      dend = AdaptEndIndex(d_offset, input_depth, output_depth);

      hstart = AdaptStartIndex(h_offset, input_height, output_height);
      hend = AdaptEndIndex(h_offset, input_height, output_height);

      wstart = AdaptStartIndex(w_offset, input_width, output_width);
      wend = AdaptEndIndex(w_offset, input_width, output_width);
    } else {
      dstart = d_offset * stride_depth - padding_depth;
      hstart = h_offset * stride_height - padding_height;
      wstart = w_offset * stride_width - padding_width;
      dend = min(dstart + ksize_depth, input_depth);
      hend = min(hstart + ksize_height, input_height);
      wend = min(wstart + ksize_width, input_width);
      dstart = max(dstart, 0);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
    }

    T1 ele = -FLT_MAX;
    int max_index = -1;
    for (int d = dstart; d < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          // 跟return_mask==false的正向的区别也就在于，把max的逻辑放进来而不是在pool_process里，从而得到index
          if (ele < input_data[(d * input_height + h) * input_width + w]) {
            max_index = (d * input_height + h) * input_width + w;
            ele = input_data[max_index];
          }
        }
      }
    }
    output_data[index] = ele;
    mask_data[index] = max_index;
  }
}
*/

// SUB:DOING 三维线程配置版的maxpool3d前向kernel
template <typename T1, typename T2>
__global__ void KernelMaxPool3DWithIdx(const int ncd,
                                       const T1* input_data,
                                       const int channels,
                                       const int input_depth,
                                       const int input_height,
                                       const int input_width,
                                       const int output_depth,
                                       const int output_height,
                                       const int output_width,
                                       const int ksize_depth,
                                       const int ksize_height,
                                       const int ksize_width,
                                       const int stride_depth,
                                       const int stride_height,
                                       const int stride_width,
                                       const int padding_depth,
                                       const int padding_height,
                                       const int padding_width,
                                       bool adaptive,
                                       T1* output_data,
                                       T2* mask_data,
                                       FastDivModForPooling3D divmods_output) {
  int w_offset, h_offset, d_offset, nc_offset;
  int dstart, dend, hstart, hend, wstart, wend;
  const T1* input_data_cur;

  w_offset = blockIdx.x * blockDim.x + threadIdx.x;
  h_offset = blockIdx.y * blockDim.y + threadIdx.y;

  if (w_offset < output_width && h_offset < output_height) {
    for (int index_z = blockIdx.z * blockDim.z + threadIdx.z; index_z < ncd; index_z += gridDim.z * blockDim.z) {
      auto output_depth_divmod = divmods_output.depth.Divmod(index_z);
      d_offset = output_depth_divmod.val[1];
      nc_offset = output_depth_divmod.val[0];
      int output_index = nc_offset * output_depth * output_height * output_width + d_offset * output_height * output_width + h_offset * output_width + w_offset;
      int input_offset = nc_offset * input_depth * input_height * input_width;
      input_data_cur = input_data + input_offset;

      if (adaptive) {
        dstart = AdaptStartIndex(d_offset, input_depth, output_depth);
        dend = AdaptEndIndex(d_offset, input_depth, output_depth);
  
        hstart = AdaptStartIndex(h_offset, input_height, output_height);
        hend = AdaptEndIndex(h_offset, input_height, output_height);
  
        wstart = AdaptStartIndex(w_offset, input_width, output_width);
        wend = AdaptEndIndex(w_offset, input_width, output_width);
      } else {
        dstart = d_offset * stride_depth - padding_depth;
        hstart = h_offset * stride_height - padding_height;
        wstart = w_offset * stride_width - padding_width;
        dend = min(dstart + ksize_depth, input_depth);
        hend = min(hstart + ksize_height, input_height);
        wend = min(wstart + ksize_width, input_width);
        dstart = max(dstart, 0);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
      }

      T1 ele = -FLT_MAX;
      int max_index = -1;
      for (int d = dstart; d < dend; ++d) {
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            if (ele < input_data_cur[(d * input_height + h) * input_width + w]) {
              max_index = (d * input_height + h) * input_width + w;
              ele = input_data_cur[max_index];
            }
          }
        }
      }
      output_data[output_index] = ele;
      mask_data[output_index] = max_index;
    }
  }
}

// SUB:DONE maxpool3d反向kernel
/*
template <typename T1, typename T2>
__global__ void KernelMaxPool3DWithIdxGrad(const int nthreads,
                                           const T1* output_grad,
                                           const T2* mask,
                                           const int channels,
                                           const int input_depth,
                                           const int input_height,
                                           const int input_width,
                                           const int output_depth,
                                           const int output_height,
                                           const int output_width,
                                           const int ksize_depth,
                                           const int ksize_height,
                                           const int ksize_width,
                                           const int stride_depth,
                                           const int stride_height,
                                           const int stride_width,
                                           const int padding_depth,
                                           const int padding_height,
                                           const int padding_width,
                                           bool adaptive,
                                           T1* input_grad,
                                           FastDivModForPooling3D divmods,
                                           FastDivModForPooling3DStride divmods_stride) {
  // input参数和stride参数其实不传都可以，因为通过divmods已经传进来了
  // 注意这个是反向，maxpooling的反向是，如果input正好等于output的值，在这个input位置上的梯度就是保持不变，其他的都是0
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    // ncdhw
    // 对于反向而言，input是output，output是input
    // 不知道为啥没有跟其他接口一样考虑ndhwc格式的索引计算？with_index的都没考虑
    // 优化索引计算中的除法和求模，通过fastdivmod计算（前面的函数参数和后面的变量名也需要修改）

    // 注意思考这里的索引计算方式，它的offset和p实际上针对的是grad_input即output来算的
    // 首先这里的grid和block是按照input来切分的，output的多个位置都可能对应到input的某个位置，所以就是循环这些位置，求和来对应到那个input位置的值
    // offset其实也是input的offset

    int pdstart, pdend, phstart, phend, pwstart, pwend;
    int w_offset, h_offset, d_offset, c_offset, output_offset;
    // 注意这里传到output索引并不是说求模的是output哦，实际上求模的是创建divmods是的input索引，output索引是用于计算output_offset
    OffsetPreparationFor5Dimension<FastDivModForPooling3D>(index, 
                                                           false, 
                                                           divmods, 
                                                           0, 
                                                           0,
                                                           0,
                                                           output_width, 
                                                           output_height, 
                                                           output_depth, 
                                                           &w_offset, 
                                                           &h_offset, 
                                                           &d_offset, 
                                                           &c_offset, 
                                                           &output_offset);
    // 后面是索引dhw，所以先把nc给偏移出去
    mask += output_offset;
    output_grad += output_offset;

    // 调用的时候是关掉adapative的，但adaptive是什么？意思就是不需要根据padding，stride，直接就根据input和output算出来kernel的作用范围
    // 这里不知道为什么没有像其他kernel一样直接用封装好的adaptive_index计算接口，不过这个应该也没什么好优化的
    // SUB:TODO adaptive部分也可以用fastdivmod优化下
    if (adaptive) {
      pdstart = d_offset * output_depth / input_depth;
      pdend =
          min((d_offset + 1) * output_depth / input_depth + 1, output_depth);
      phstart = h_offset * output_height / input_height;
      phend =
          min((h_offset + 1) * output_height / input_height + 1, output_height);
      pwstart = w_offset * output_width / input_width;
      pwend =
          min((w_offset + 1) * output_width / input_width + 1, output_width);
    } else {
      // 计算offset的时候，其实锚定原始的input网格，现在是根据对input的padding和kernel_size和stride，重新锚定到output的网格，从移动的视角
      // 用fastdivmod优化这一部分除法，三目运算符就是语法糖，并不是说有更好的性能，可以放心优化
      if (d_offset + padding_depth < ksize_depth) 
        pdstart = 0;
      else 
        pdstart = divmods_stride.depth.Div(d_offset + padding_depth - ksize_depth) + 1;
      
      if (h_offset + padding_height < ksize_height) 
        phstart = 0;
      else
        phstart = divmods_stride.height.Div(h_offset + padding_height - ksize_height) + 1;
  
      if (w_offset + padding_width < ksize_width)
        pwstart = 0;
      else
        pwstart = divmods_stride.width.Div(w_offset + padding_width - ksize_width) + 1;
      
      pdend = min(divmods_stride.depth.Div(d_offset + padding_depth) + 1, output_depth);
      phend = min(divmods_stride.height.Div(h_offset + padding_height) + 1, output_height);
      pwend = min(divmods_stride.width.Div(w_offset + padding_width) + 1, output_width);
    }

    T1 input_grad_data = 0;
    int input_current_feature_map_idx =
        (d_offset * input_height + h_offset) * input_width + w_offset;
    for (int pd = pdstart; pd < pdend; ++pd) {
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // mask存的是跟output大小一样的，对应到input的最大索引
          if (mask[(pd * output_height + ph) * output_width + pw] ==
              input_current_feature_map_idx)
            input_grad_data +=
                output_grad[(pd * output_height + ph) * output_width + pw];
        }
      }
    }
    input_grad[index] = input_grad_data;
  }
}
*/

// SUB:DONE 三维线程配置版的maxpool3d反向kernel
/*
template <typename T1, typename T2>
__global__ void KernelMaxPool3DWithIdxGrad(const int ncd,
                                           const T1* output_grad,
                                           const T2* mask,
                                           const int channels,
                                           const int input_depth,
                                           const int input_height,
                                           const int input_width,
                                           const int output_depth,
                                           const int output_height,
                                           const int output_width,
                                           const int ksize_depth,
                                           const int ksize_height,
                                           const int ksize_width,
                                           const int stride_depth,
                                           const int stride_height,
                                           const int stride_width,
                                           const int padding_depth,
                                           const int padding_height,
                                           const int padding_width,
                                           bool adaptive,
                                           T1* input_grad,
                                           FastDivModForPooling3D divmods_input,
                                           FastDivModForPooling3DStride divmods_stride) {
  // input参数和stride参数其实不传都可以，因为通过divmods已经传进来了
  // 注意这个是反向，maxpooling的反向是，如果input正好等于output的值，在这个input位置上的梯度就是保持不变，其他的都是0

  // 注意思考这里的索引计算方式，它的offset和p实际上针对的是grad_input即output来算的
  // 首先这里的grid和block是按照input来切分的，output的多个位置都可能对应到input的某个位置，所以就是循环这些位置，求和来对应到那个input位置的值
  // offset其实也是input的offset

  int w_offset, h_offset, d_offset, c_offset, output_offset; 
  int pdstart, pdend, phstart, phend, pwstart, pwend;

  w_offset = blockIdx.x * blockDim.x + threadIdx.x;
  h_offset = blockIdx.y * blockDim.y + threadIdx.y;

  // 一直没有注意到这一点，这里要注意线程配置的时候会向2次幂取整，但实际index的时候不能越界
  if (w_offset < input_width && h_offset < input_height) {
    // SUB:TODO 这里还可以减少除法和求模次数
    // 这样多次循环会不会造成除法次数太多呢？我可以再抽出一个stride计算，每次offset+stride就好，但只循环一次的情况反而更慢
    for (int index_z = blockIdx.z * blockDim.z + threadIdx.z; index_z < ncd; index_z += gridDim.z * blockDim.z) {
        auto input_depth_divmod = divmods_input.depth.Divmod(index_z);
        auto channel_divmod = divmods_input.channel.Divmod(input_depth_divmod.val[0]);
        // SUB:TODO 这里其实不需要计算c_offset，直接nc_offset就好
        d_offset = input_depth_divmod.val[1];
        c_offset = channel_divmod.val[1];
        output_offset = (channel_divmod.val[0] * divmods_input.channel.divisor + c_offset) * output_depth * output_height * output_width;

        if (adaptive) {
          pdstart = divmods_input.depth.Div(d_offset * output_depth);
          pdend = min(divmods_input.depth.Div((d_offset + 1) * output_depth) + 1, output_depth);
          phstart = divmods_input.height.Div(h_offset * output_height);
          phend = min(divmods_input.height.Div((h_offset + 1) * output_height) + 1, output_height);
          pwstart = divmods_input.width.Div(w_offset * output_width);
          pwend = min(divmods_input.width.Div((w_offset + 1) * output_width) + 1, output_width);
        } else {
          // 计算offset的时候，其实锚定原始的input网格，现在是根据对input的padding和kernel_size和stride，重新锚定到output的网格，从移动的视角
          pdstart = (d_offset + padding_depth < ksize_depth) ? 0 : divmods_stride.depth.Div(d_offset + padding_depth - ksize_depth) + 1;
          phstart = (h_offset + padding_height < ksize_height) ? 0 : divmods_stride.height.Div(h_offset + padding_height - ksize_height) + 1;
          pwstart = (w_offset + padding_width < ksize_width) ? 0 : divmods_stride.width.Div(w_offset + padding_width - ksize_width) + 1;
          pdend = min(divmods_stride.depth.Div(d_offset + padding_depth) + 1, output_depth);
          phend = min(divmods_stride.height.Div(h_offset + padding_height) + 1, output_height);
          pwend = min(divmods_stride.width.Div(w_offset + padding_width) + 1, output_width);
        }

        T1 input_grad_data = 0;
        int input_current_feature_map_idx =
            (d_offset * input_height + h_offset) * input_width + w_offset;
        for (int pd = pdstart; pd < pdend; ++pd) {
          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              // mask存的是跟output大小一样的，对应到input的最大索引
              // 这里对output_offset的处理也是，实际上是相对最开始位置的offset，而不是每次循环的stride，所以不能用+=
              if (mask[output_offset + (pd * output_height + ph) * output_width + pw] ==
                  input_current_feature_map_idx)
                input_grad_data +=
                    output_grad[output_offset + (pd * output_height + ph) * output_width + pw];
            }
          }
        }
        input_grad[(index_z * input_height + h_offset) * input_width + w_offset] = input_grad_data;
    }
  }
}
*/

// SUB:DONE pytorch版的maxpool3d反向kernel
template <typename T1, typename T2>
__global__ void KernelMaxPool3DWithIdxGrad(const int ncd,
                                           const T1* output_grad,
                                           const T2* mask,
                                           const int channels,
                                           const int input_depth,
                                           const int input_height,
                                           const int input_width,
                                           const int output_depth,
                                           const int output_height,
                                           const int output_width,
                                           const int ksize_depth,
                                           const int ksize_height,
                                           const int ksize_width,
                                           const int stride_depth,
                                           const int stride_height,
                                           const int stride_width,
                                           const int padding_depth,
                                           const int padding_height,
                                           const int padding_width,
                                           bool adaptive,
                                           T1* input_grad,
                                           FastDivModForPooling3D divmods_output) {
  int w_offset, h_offset, d_offset, nc_offset; 

  w_offset = blockIdx.x * blockDim.x + threadIdx.x;
  h_offset = blockIdx.y * blockDim.y + threadIdx.y;

  // 一直没有注意到这一点，这里要注意线程配置的时候会向2次幂取整，但实际index的时候不能越界
  if (w_offset < output_width && h_offset < output_height) {
    for (int index_z = blockIdx.z * blockDim.z + threadIdx.z; index_z < ncd; index_z += gridDim.z * blockDim.z) {
      auto output_depth_divmod = divmods_output.depth.Divmod(index_z);
      d_offset = output_depth_divmod.val[1];
      nc_offset = output_depth_divmod.val[0];
      int output_index = nc_offset * output_depth * output_height * output_width + d_offset * output_height * output_width + h_offset * output_width + w_offset;
      int max_index = mask[output_index];
      if (max_index != -1) {
        paddle::platform::CudaAtomicAdd(&input_grad[nc_offset * input_depth * input_height * input_width + max_index], output_grad[output_index]);
      }
    }
  }
}

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
 // SUB:DONE maxpool3d起前向
 /*
template <typename T1, typename T2>
class MaxPool3dWithIndexFunctor<phi::GPUContext, T1, T2> {
 public:
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool adaptive,
                  DenseTensor* output,
                  DenseTensor* mask) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output->dims()[1];
    const int output_depth = output->dims()[2];
    const int output_height = output->dims()[3];
    const int output_width = output->dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T1* input_data = input.data<T1>();
    // 这是什么语法，没看懂
    T1* output_data = context.template Alloc<T1>(output);
    T2* mask_data = context.template Alloc<T2>(mask);

    // 按照output元素的数量设置的线程数
    int nthreads = batch_size * output_channels * output_depth * output_height *
                   output_width;
    int thread_num = 1024;
#ifdef WITH_NV_JETSON
    backends::gpu::ChangeThreadNum(context, &thread_num);
#endif

    // 理论上需要更通用的计算方式，但考虑到测例不多，而且每个block尽可能多线程可以减少切换开销，所以这里的优化空间可能并不大
    int blocks = (nthreads + thread_num - 1) / thread_num;
    dim3 threads(thread_num, 1);
    dim3 grid(blocks, 1);

    // pool_divmods传到前向kernel
    auto pool_divmods = FastDivModForPooling3D(input_channels, output_width, output_height, output_depth);

    KernelMaxPool3DWithIdx<T1, T2>
        <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                 input_data,
                                                 input_channels,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_depth,
                                                 output_height,
                                                 output_width,
                                                 ksize_depth,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_depth,
                                                 stride_height,
                                                 stride_width,
                                                 padding_depth,
                                                 padding_height,
                                                 padding_width,
                                                 adaptive,
                                                 output_data,
                                                 mask_data,
                                                 pool_divmods);
  }
};
*/

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
 // SUB:DOING 三维线程配置版的maxpool3d起前向
 template <typename T1, typename T2>
 class MaxPool3dWithIndexFunctor<phi::GPUContext, T1, T2> {
  public:
   void operator()(const phi::GPUContext& context,
                   const DenseTensor& input,
                   const std::vector<int>& ksize,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   bool adaptive,
                   DenseTensor* output,
                   DenseTensor* mask) {
     const int batch_size = input.dims()[0];
     const int input_channels = input.dims()[1];
     const int input_depth = input.dims()[2];
     const int input_height = input.dims()[3];
     const int input_width = input.dims()[4];
     const int output_channels = output->dims()[1];
     const int output_depth = output->dims()[2];
     const int output_height = output->dims()[3];
     const int output_width = output->dims()[4];
     const int ksize_depth = ksize[0];
     const int ksize_height = ksize[1];
     const int ksize_width = ksize[2];
     const int stride_depth = strides[0];
     const int stride_height = strides[1];
     const int stride_width = strides[2];
     const int padding_depth = paddings[0];
     const int padding_height = paddings[1];
     const int padding_width = paddings[2];
 
     const T1* input_data = input.data<T1>();
     T1* output_data = context.template Alloc<T1>(output);
     T2* mask_data = context.template Alloc<T2>(mask);
 
     int ncd = batch_size * input_channels * output_depth;
 
    //  backends::gpu::GpuLaunchConfig config = backends::gpu::GetGpuLaunchConfig3D(context, ncd, output_height, output_width);
    //  dim3 threads = config.thread_per_block;
    //  dim3 grid = config.block_per_grid;

     int thread_x = 32;
     int thread_y = 8;
     int thread_z = 1;
     dim3 threads(thread_x, thread_y, thread_z);
     std::array<int, 3> max_grid_dim = context.GetCUDAMaxGridDimSize();
     int block_x = (output_width + threads.x - 1) / threads.x;
     int block_y = (output_height + threads.y - 1) / threads.y;
     int block_z = (ncd > max_grid_dim[2] * threads.z) ? max_grid_dim[2] : (ncd + threads.z - 1) / threads.z;
     dim3 grid(block_x, block_y, block_z);
 
     auto pool_divmods_output = FastDivModForPooling3D(input_channels, output_width, output_height, output_depth);
 
     KernelMaxPool3DWithIdx<T1, T2>
         <<<grid, threads, 0, context.stream()>>>(ncd,
                                                  input_data,
                                                  input_channels,
                                                  input_depth,
                                                  input_height,
                                                  input_width,
                                                  output_depth,
                                                  output_height,
                                                  output_width,
                                                  ksize_depth,
                                                  ksize_height,
                                                  ksize_width,
                                                  stride_depth,
                                                  stride_height,
                                                  stride_width,
                                                  padding_depth,
                                                  padding_height,
                                                  padding_width,
                                                  adaptive,
                                                  output_data,
                                                  mask_data,
                                                  pool_divmods_output);
   }
 };

/*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
 // SUB:DONE maxpool3d起反向
 /*
 template <typename T1, typename T2>
 class MaxPool3dWithIndexGradFunctor<phi::GPUContext, T1, T2> {
  public:
   void operator()(const phi::GPUContext& context,
                   const DenseTensor& output_grad,
                   const DenseTensor& mask,
                   const std::vector<int>& ksize,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   bool adaptive,
                   DenseTensor* input_grad) {
     const int batch_size = input_grad->dims()[0];
     const int input_channels = input_grad->dims()[1];
     const int input_depth = input_grad->dims()[2];
     const int input_height = input_grad->dims()[3];
     const int input_width = input_grad->dims()[4];
     const int output_depth = output_grad.dims()[2];
     const int output_height = output_grad.dims()[3];
     const int output_width = output_grad.dims()[4];
     const int ksize_depth = ksize[0];
     const int ksize_height = ksize[1];
     const int ksize_width = ksize[2];
     const int stride_depth = strides[0];
     const int stride_height = strides[1];
     const int stride_width = strides[2];
     const int padding_depth = paddings[0];
     const int padding_height = paddings[1];
     const int padding_width = paddings[2];
 
     const T1* output_grad_data = output_grad.data<T1>();
     const T2* mask_data = mask.data<T2>();
     // 怪怪的，看起来是alloc，但不是应该copy吗
     T1* input_grad_data = context.template Alloc<T1>(input_grad);

     int nthreads =
         batch_size * input_channels * input_depth * input_height * input_width;
     int blocks = (nthreads + 1024 - 1) / 1024;
     dim3 threads(1024, 1);
     dim3 grid(blocks, 1);

     // pool_divmods传到反向kernel
     auto pool_divmods = FastDivModForPooling3D(input_channels, input_width, input_height, input_depth);

     // pool_stride_divmods传到反向kernel
     auto pool_stride_divmods = FastDivModForPooling3DStride(stride_width, stride_height, stride_depth);
 
     KernelMaxPool3DWithIdxGrad<T1, T2>
         <<<grid, threads, 0, context.stream()>>>(nthreads,
                                                  output_grad_data,
                                                  mask_data,
                                                  input_channels,
                                                  input_depth,
                                                  input_height,
                                                  input_width,
                                                  output_depth,
                                                  output_height,
                                                  output_width,
                                                  ksize_depth,
                                                  ksize_height,
                                                  ksize_width,
                                                  stride_depth,
                                                  stride_height,
                                                  stride_width,
                                                  padding_depth,
                                                  padding_height,
                                                  padding_width,
                                                  adaptive,
                                                  input_grad_data,
                                                  pool_divmods,
                                                  pool_stride_divmods);
   }
 };
 */

 /*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
 // SUB:DONE 三维线程配置版的maxpool3d起反向
 /*
template <typename T1, typename T2>
class MaxPool3dWithIndexGradFunctor<phi::GPUContext, T1, T2> {
 public:
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& output_grad,
                  const DenseTensor& mask,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  bool adaptive,
                  DenseTensor* input_grad) {
    const int batch_size = input_grad->dims()[0];
    const int input_channels = input_grad->dims()[1];
    const int input_depth = input_grad->dims()[2];
    const int input_height = input_grad->dims()[3];
    const int input_width = input_grad->dims()[4];
    const int output_depth = output_grad.dims()[2];
    const int output_height = output_grad.dims()[3];
    const int output_width = output_grad.dims()[4];
    const int ksize_depth = ksize[0];
    const int ksize_height = ksize[1];
    const int ksize_width = ksize[2];
    const int stride_depth = strides[0];
    const int stride_height = strides[1];
    const int stride_width = strides[2];
    const int padding_depth = paddings[0];
    const int padding_height = paddings[1];
    const int padding_width = paddings[2];

    const T1* output_grad_data = output_grad.data<T1>();
    const T2* mask_data = mask.data<T2>();
    T1* input_grad_data = context.template Alloc<T1>(input_grad);

    int ncd = batch_size * input_channels * input_depth;

    backends::gpu::GpuLaunchConfig config = backends::gpu::GetGpuLaunchConfig3D(context, ncd, input_height, input_width);
    dim3 threads = config.thread_per_block;
    dim3 grid = config.block_per_grid;

    // int thread_x = 32;
    // int thread_y = 8;
    // int thread_z = 1;
    // dim3 threads(thread_x, thread_y, thread_z);
    // std::array<int, 3> max_grid_dim = context.GetCUDAMaxGridDimSize();
    // int block_x = (input_width + threads.x - 1) / threads.x;
    // int block_y = (input_height + threads.y - 1) / threads.y;
    // int block_z = (ncd > max_grid_dim[2] * threads.z) ? max_grid_dim[2] : (ncd + threads.z - 1) / threads.z;
    // dim3 grid(block_x, block_y, block_z);
      
    auto pool_divmods_input = FastDivModForPooling3D(input_channels, input_width, input_height, input_depth);
    auto pool_divmods_stride = FastDivModForPooling3DStride(stride_width, stride_height, stride_depth);

    KernelMaxPool3DWithIdxGrad<T1, T2>
        <<<grid, threads, 0, context.stream()>>>(ncd,
                                                 output_grad_data,
                                                 mask_data,
                                                 input_channels,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_depth,
                                                 output_height,
                                                 output_width,
                                                 ksize_depth,
                                                 ksize_height,
                                                 ksize_width,
                                                 stride_depth,
                                                 stride_height,
                                                 stride_width,
                                                 padding_depth,
                                                 padding_height,
                                                 padding_width,
                                                 adaptive,
                                                 input_grad_data,
                                                 pool_divmods_input,
                                                 pool_divmods_stride);
  }
};
*/

 /*
 * All tensors are in NCDHW format.
 * Ksize, strides, paddings are three elements. These three elements represent
 * depth, height and width, respectively.
 */
 // SUB:DONE pytorch版的maxpool3d起反向
 template <typename T1, typename T2>
 class MaxPool3dWithIndexGradFunctor<phi::GPUContext, T1, T2> {
  public:
   void operator()(const phi::GPUContext& context,
                   const DenseTensor& output_grad,
                   const DenseTensor& mask,
                   const std::vector<int>& ksize,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   bool adaptive,
                   DenseTensor* input_grad) {
     const int batch_size = input_grad->dims()[0];
     const int input_channels = input_grad->dims()[1];
     const int input_depth = input_grad->dims()[2];
     const int input_height = input_grad->dims()[3];
     const int input_width = input_grad->dims()[4];
     const int output_depth = output_grad.dims()[2];
     const int output_height = output_grad.dims()[3];
     const int output_width = output_grad.dims()[4];
     const int ksize_depth = ksize[0];
     const int ksize_height = ksize[1];
     const int ksize_width = ksize[2];
     const int stride_depth = strides[0];
     const int stride_height = strides[1];
     const int stride_width = strides[2];
     const int padding_depth = paddings[0];
     const int padding_height = paddings[1];
     const int padding_width = paddings[2];
 
     const T1* output_grad_data = output_grad.data<T1>();
     const T2* mask_data = mask.data<T2>();
     T1* input_grad_data = context.template Alloc<T1>(input_grad);
 
     int ncd = batch_size * input_channels * output_depth;
 
    //  backends::gpu::GpuLaunchConfig config = backends::gpu::GetGpuLaunchConfig3D(context, ncd, output_height, output_width);
    //  dim3 threads = config.thread_per_block;
    //  dim3 grid = config.block_per_grid;
 
     int thread_x = 32;
     int thread_y = 8;
     int thread_z = 1;
     dim3 threads(thread_x, thread_y, thread_z);
     std::array<int, 3> max_grid_dim = context.GetCUDAMaxGridDimSize();
     int block_x = (output_width + threads.x - 1) / threads.x;
     int block_y = (output_height + threads.y - 1) / threads.y;
     int block_z = (ncd > max_grid_dim[2] * threads.z) ? max_grid_dim[2] : (ncd + threads.z - 1) / threads.z;
     dim3 grid(block_x, block_y, block_z);
       
     auto pool_divmods_output = FastDivModForPooling3D(input_channels, output_width, output_height, output_depth);
 
     KernelMaxPool3DWithIdxGrad<T1, T2>
         <<<grid, threads, 0, context.stream()>>>(ncd,
                                                  output_grad_data,
                                                  mask_data,
                                                  input_channels,
                                                  input_depth,
                                                  input_height,
                                                  input_width,
                                                  output_depth,
                                                  output_height,
                                                  output_width,
                                                  ksize_depth,
                                                  ksize_height,
                                                  ksize_width,
                                                  stride_depth,
                                                  stride_height,
                                                  stride_width,
                                                  padding_depth,
                                                  padding_height,
                                                  padding_width,
                                                  adaptive,
                                                  input_grad_data,
                                                  pool_divmods_output);
   }
 };

template class MaxPool3dWithIndexFunctor<phi::GPUContext, float, int>;
template class MaxPool3dWithIndexGradFunctor<phi::GPUContext, float, int>;
template class MaxPool3dWithIndexFunctor<phi::GPUContext, double, int>;
template class MaxPool3dWithIndexGradFunctor<phi::GPUContext, double, int>;

}  // namespace funcs
}  // namespace phi
