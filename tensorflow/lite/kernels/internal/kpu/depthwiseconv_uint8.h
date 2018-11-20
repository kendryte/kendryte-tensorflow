/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_KPU_DEPTHWISECONV_UINT8_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_KPU_DEPTHWISECONV_UINT8_H_

#include <algorithm>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "hal.h"
#include "printf.h"
#include "kpu_param.h"

namespace tflite {
namespace kpu_ops {

template<class T>
inline uint64_t to_ui64(const T& reg)
{
    union
    {
        T reg;
        uint64_t data;
    } u;
    u.reg = reg;
    return u.data;
}

#define KPU_DEBUG 0

inline void DepthwiseConv(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data) {
  const uint32_t stride_width = params.stride_width;
  const uint32_t stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int depth_multiplier = params.depth_multiplier;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const uint32_t output_shift = params.output_shift;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const uint32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  const uint32_t output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const uint32_t input_height = input_shape.Dims(1);
  const uint32_t input_width = input_shape.Dims(2);
  const uint32_t input_depth = input_shape.Dims(3);
  const uint32_t filter_height = filter_shape.Dims(1);
  const uint32_t filter_width = filter_shape.Dims(2);
  const uint32_t output_height = output_shape.Dims(1);
  const uint32_t output_width = output_shape.Dims(2);
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

#if 1
  uint32_t in_row_padding;
  uint32_t in_row_group;
  uint32_t in_row_length;

  if (input_width <= 16) {
    in_row_padding = 16;
    in_row_group = 4;
    in_row_length = 1;
  }
  else if (input_width <= 32) {
    in_row_padding = 32;
    in_row_group = 2;
    in_row_length = 1;
  }
  else {
    in_row_padding = 64;
    in_row_group = 1;
    in_row_length = (input_width + 63) / 64;
  }
  
  uint32_t out_row_padding;
  uint32_t out_row_group;
  uint32_t out_row_length;

  if (output_width <= 16) {
    out_row_padding = 16;
    out_row_group = 4;
    out_row_length = 1;
  }
  else if (output_width <= 32) {
    out_row_padding = 32;
    out_row_group = 2;
    out_row_length = 1;
  }
  else {
    out_row_padding = 64;
    out_row_group = 1;
    out_row_length = (output_width + 63) / 64;
  }

  const uint32_t in_channels_of_group = std::min(input_depth, in_row_group);
  const uint32_t out_channels_of_group = std::min(output_depth, out_row_group);
  const uint32_t out_channel_kernel_size = filter_width * filter_height * input_depth;

#if KPU_DEBUG
  printk("kpu depthconv: x %d >> %d, i: %dx%d o:%dx%d, %d\n", output_multiplier, -output_shift, input_width, input_height, output_width, output_height, out_channel_kernel_size);
  printk("io: %d, oo: %d\n ", input_offset, output_offset);
#endif
  
  struct timeval tv, tv2;
  gettimeofday(&tv, NULL);
  kpu_layer_argument_t layer;
  layer.interrupt_enabe.data = {
     .int_en = 1,
     .ram_flag = 0,
     .full_add = 0,
     .depth_wise_layer = 1
  };
  layer.image_addr.data = {
     .image_src_addr = (uint64_t)0x0,
     .image_dst_addr = (uint64_t)(0x8000 - (64 * in_row_length * input_height * output_depth / out_channels_of_group + 63) / 64)
  };
  layer.image_channel_num.data = {
     .i_ch_num = input_depth - 1,
     .o_ch_num = output_depth - 1,
     .o_ch_num_coef = output_depth - 1
  };
  layer.image_size.data = {
     .i_row_wid = input_width - 1,
     .i_col_high = input_height - 1,
     .o_row_wid = input_width - 1,
     .o_col_high = input_height - 1
  };
  layer.kernel_pool_type_cfg.data = {
     .kernel_type = filter_width == 3 ? 1U : 0,
     .pad_type = 0,
     .pool_type = 0,
     .first_stride = 0,
     .bypass_conv = 0,
     .load_para = 1,
     .dma_burst_size = 15,
     .pad_value = 0,
     .bwsx_base_addr = 0
  };
  layer.kernel_load_cfg.data = {
     .load_coor = 1,
     .load_time = 0,
     .para_size = out_channel_kernel_size,
     .para_start_addr = 0
  };
  layer.kernel_offset.data = {
     .coef_column_offset = 0,
     .coef_row_offset = 0
  };
  layer.kernel_calc_type_cfg.data = {
     .channel_switch_addr = in_row_length * input_height,
     .row_switch_addr = in_row_length,
     .coef_size = 0,
     .coef_group = in_row_group,
     .load_act = 1,
     .active_addr = 0
  };
  layer.write_back_cfg.data = {
     .wb_channel_switch_addr = in_row_length * input_height,
     .wb_row_switch_addr = in_row_length,
     .wb_group = in_row_group
  };
  layer.conv_value.data = {
     .shr_w = 0,
     .shr_x = 0,
     .arg_w = static_cast<uint64_t>(input_offset),
     .arg_x = static_cast<uint64_t>(filter_offset)
  };
  layer.conv_value2.data = {
     .arg_add = static_cast<uint64_t>(input_offset * filter_offset * filter_width * filter_height)
  };
  layer.dma_parameter.data = {
     .send_data_out = 1,
     .channel_byte_num = input_width * input_height - 1,
     .dma_total_byte = (input_width * input_height * output_depth) - 1
  };

  auto kpu_bn_table = std::make_unique<kpu_batchnorm_argument_t[]>(output_depth);
  uint64_t mul = output_multiplier >> (12 - output_shift);
  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    int64_t add = bias_data ? bias_data[out_channel] : 0;
    add = (add * mul) >> 15;
    add += output_offset << 4;
    kpu_bn_table[out_channel].batchnorm.data = {.norm_mul = mul,.norm_add = static_cast<uint64_t>(add),.norm_shift = 15 };
  }

  auto kernels = std::make_unique<uint8_t[]>(filter_width * filter_height * input_depth);
  auto inputs = reinterpret_cast<uint8_t*>(AI_IO_BASE_ADDR);

  size_t output_size = ((layer.dma_parameter.data.dma_total_byte + 1) + 7) / 8 * 8;
  auto outputs = std::make_unique<uint8_t[]>(output_size);

  layer.kernel_pool_type_cfg.data.bwsx_base_addr = (uint64_t)kpu_bn_table.get();
  layer.kernel_calc_type_cfg.data.active_addr = (uint64_t)&kpu_act_table;
  layer.kernel_load_cfg.data.para_start_addr = (uint64_t)kernels.get();

  // init act
  kpu_act_table.activate_para[0].data =  { .shift_number = 0,.y_mul = 0,.x_start = 0x800000000 };
  kpu_act_table.activate_para[1].data =  { .shift_number = 0,.y_mul = 0,.x_start = 0xf7d4cf4b8 };
  kpu_act_table.activate_para[2].data =  { .shift_number = 0,.y_mul = 0,.x_start = 0xf8ed5a20c };
  kpu_act_table.activate_para[3].data =  { .shift_number = 0,.y_mul = 0,.x_start = 0xfa05e4f60 };
  kpu_act_table.activate_para[4].data =  { .shift_number = 0,.y_mul = 0,.x_start = 0xfb2e05baa };
  kpu_act_table.activate_para[5].data =  { .shift_number = 0,.y_mul = 0,.x_start = 0xfc46908fe };
  kpu_act_table.activate_para[6].data =  { .shift_number = 0,.y_mul = 0,.x_start = 0xfd5f1b652 };
  kpu_act_table.activate_para[7].data =  { .shift_number = 0,.y_mul = 0,.x_start = 0xfe77a63a6 };
  kpu_act_table.activate_para[8].data =  { .shift_number = 0,.y_mul = 0,.x_start = 0xff9fc6ff0 };
  kpu_act_table.activate_para[9].data =  { .shift_number = 0,.y_mul = 0,.x_start = 0xfffd4a9b7 };
  kpu_act_table.activate_para[10].data = { .shift_number = 4,.y_mul = 1,.x_start = 0 };
  kpu_act_table.activate_para[11].data = { .shift_number = 0,.y_mul = 0,.x_start = 0x1d0dca98  };
  kpu_act_table.activate_para[12].data = { .shift_number = 0,.y_mul = 0,.x_start = 0x2e9677ec  };
  kpu_act_table.activate_para[13].data = { .shift_number = 0,.y_mul = 0,.x_start = 0x401f253f  };
  kpu_act_table.activate_para[14].data = { .shift_number = 0,.y_mul = 0,.x_start = 0x52a1318a  };
  kpu_act_table.activate_para[15].data = { .shift_number = 0,.y_mul = 0,.x_start = 0x6429dedd  };
  kpu_act_table.activate_para_bias0.data = {
    .result_bias = {0,0,0,0,0,0,0,0}
  };
  kpu_act_table.activate_para_bias1.data = {
    .result_bias = {0,0,0,0,0,0,0,0}
  };

  // init kernels
  {
    uint8_t *k_it = kernels.get();
    for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
      for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
          *k_it++ = filter_data[Offset(filter_shape, 0, filter_y,
              filter_x, in_channel)];
        }
      }
    }
#if KPU_DEBUG
    printk("kernels\n");
    for (size_t i = 0; i < 64; i++)
        printk("%d ", kernels[i]);
#endif
  }

  for (int batch = 0; batch < batches; ++batch) {
    // init inputs
    {
      for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
        auto channel_origin = inputs + in_channel / in_row_group * in_row_length * input_height * 64 + in_channel % in_row_group * in_row_padding;
        for (int in_y = 0; in_y < input_height; ++in_y) {
          auto y_origin = channel_origin + in_y * in_row_length * 64;
          for (int in_x = 0; in_x < input_width; ++in_x) {
              y_origin[in_x] = input_data[Offset(input_shape, batch, in_y,
                  in_x, in_channel)];
          }
        }
      }

#if KPU_DEBUG
      printk("inputs\n");
      for (size_t i = 0; i < 64; i++)
          printk("%d ", input_data[i]);
#endif
    }

    volatile kpu_config_t *const kpu = (volatile kpu_config_t *)AI_BASE_ADDR;
    kpu->interrupt_clear.reg = to_ui64(kpu_config_interrupt_t
    {
        .calc_done_int = 1,
        .layer_cfg_almost_empty_int = 1,
        .layer_cfg_almost_full_int = 1
    });
    kpu->eight_bit_mode.reg = to_ui64(kpu_config_eight_bit_mode_t
    {
        .eight_bit_mode = 1
    });
    kpu->fifo_threshold.reg = to_ui64(kpu_config_fifo_threshold_t
    {
        .fifo_full_threshold = 10, .fifo_empty_threshold = 1
    });
    kpu->interrupt_mask.reg = to_ui64(kpu_config_interrupt_t
    {
        .calc_done_int = 0,
        .layer_cfg_almost_empty_int = 1,
        .layer_cfg_almost_full_int = 1
    });

    kpu->layer_argument_fifo = layer.interrupt_enabe.reg;
    kpu->layer_argument_fifo = layer.image_addr.reg;
    kpu->layer_argument_fifo = layer.image_channel_num.reg;
    kpu->layer_argument_fifo = layer.image_size.reg;
    kpu->layer_argument_fifo = layer.kernel_pool_type_cfg.reg;
    kpu->layer_argument_fifo = layer.kernel_load_cfg.reg;
    kpu->layer_argument_fifo = layer.kernel_offset.reg;
    kpu->layer_argument_fifo = layer.kernel_calc_type_cfg.reg;
    kpu->layer_argument_fifo = layer.write_back_cfg.reg;
    kpu->layer_argument_fifo = layer.conv_value.reg;
    kpu->layer_argument_fifo = layer.conv_value2.reg;
    kpu->layer_argument_fifo = layer.dma_parameter.reg;

    handle_t dma = dma_open_free();
    dma_set_request_source(dma, SYSCTL_DMA_SELECT_AI_RX_REQ);
    dma_transmit(dma, (void *)(&kpu->fifo_data_out), outputs.get(), false, true, 8, (layer.dma_parameter.data.dma_total_byte + 8) / 8, 8);
    dma_close(dma);

    if (stride_width == 2) {
      for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
        auto channel_origin = outputs.get() + out_channel * input_height * input_width;
        for (int out_y = 0; out_y < output_height; ++out_y) {
          auto y_origin = channel_origin + (out_y * 2 + 1) * input_width;
          for (int out_x = 0; out_x < output_width; ++out_x) {
              output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = y_origin[out_x * 2 + 1];
          }
        }
      }
    } else {
      uint8_t *o_it = outputs.get();
      for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
          for (int out_x = 0; out_x < output_width; ++out_x) {
              output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = *o_it++;
          }
        }
      }
    }
#if KPU_DEBUG
    printk("outputs: %p\n", output_data);
    for (size_t i = 0; i < 64; i++)
        printk("%d ", output_data[i]);
    gettimeofday(&tv2, NULL);
    printk("\ndepthconv used %dms.\n", (int)((tv2.tv_sec * 1000 + tv2.tv_usec / 1e3) - (tv.tv_sec * 1000 + tv.tv_usec / 1e3)));
#endif
  }
#else
  for (int b = 0; b < batches; ++b) {
#if KPU_DEBUG
      printk("depth_multiplier: %d, inputs\n", depth_multiplier);
      for (size_t i = 0; i < 64; i++)
          printk("%d ", input_data[i]);
#endif
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int ic = 0; ic < input_depth; ++ic) {
          for (int m = 0; m < depth_multiplier; m++) {
            const int oc = m + ic * depth_multiplier;
            const int in_x_origin = (out_x * stride_width) - pad_width;
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int32 acc = 0;
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y =
                    in_y_origin + dilation_height_factor * filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  int32 input_val =
                      input_data[Offset(input_shape, b, in_y, in_x, ic)];
                  int32 filter_val = filter_data[Offset(
                      filter_shape, 0, filter_y, filter_x, oc)];
                  acc +=
                      (filter_val + filter_offset) * (input_val + input_offset);
                }
              }
            }
            if (bias_data) {
              acc += bias_data[oc];
            }
            acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
                                                output_shift);
            acc += output_offset;
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);
            output_data[Offset(output_shape, b, out_y, out_x, oc)] =
                static_cast<uint8>(acc);
          }
        }
      }
    }
#if KPU_DEBUG
    printk("outputs: %p\n", output_data);
    for (size_t i = 0; i < 64; i++)
        printk("%d ", output_data[i]);
    while (1);
#endif
  }
#endif
}

}  // end namespace kpu_ops
}  // end namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_KPU_DEPTHWISECONV_UINT8_H_
