/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use standard C library call.
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <vector>

namespace tvm {
namespace contrib {

using namespace runtime;

// TVM_REGISTER_GLOBAL("tvm.contrib.quantize.mm_dequantize")
// .set_body([](TVMArgs args, TVMRetValue* ret) {
//   DLTensor* weight_ptr = args[0];
//   DLTensor* data_ptr = args[1];
//   DLTensor* w_scale_ptr = args[2];
//   DLTensor* d_scale_ptr = args[3];
//   DLTensor* w_zero_point_ptr = args[4];
//   DLTensor* d_zero_point_ptr = args[5];
//   int
// });

TVM_REGISTER_GLOBAL("tvm.contrib.quantize.choose_quantize_params")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* data_min_ptr = args[0];
  DLTensor* data_max_ptr = args[1];
  DLTensor* zero_point_ptr = args[2];
  DLTensor* scale_ptr = args[3];
  int32_t qmin = args[4];
  int32_t qmax = args[5];

  float data_min = *(static_cast<float *>(data_min_ptr->data));
  float data_max = *(static_cast<float *>(data_max_ptr->data));
  // copy from fbgemm implementation
  double scale =
        (std::max(data_max, 0.f) - std::min(data_min, 0.f)) / ((double)qmax - qmin);
  if (scale == 0) {
      scale = 0.1;
  }
  data_min = std::min(data_min, 0.f);
  data_max = std::max(data_max, 0.f);

  double zero_point_from_min = qmin - data_min / scale;
  double zero_point_from_max = qmax - data_max / scale;
  double zero_point_from_min_error = std::abs(qmin) + std::abs(data_min / scale);
  double zero_point_from_max_error = std::abs(qmax) + std::abs(data_max / scale);
  double initial_zero_point =
      zero_point_from_min_error < zero_point_from_max_error
      ? zero_point_from_min
      : zero_point_from_max;
  int32_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
   nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
   nudged_zero_point = qmax;
  } else {
   nudged_zero_point = std::nearbyint(initial_zero_point);
  }

  auto zero_point_data_ptr = static_cast<int32_t *>(zero_point_ptr->data);
  auto scale_data_ptr = static_cast<float *>(scale_ptr->data);
  *zero_point_data_ptr = nudged_zero_point;
  *scale_data_ptr = scale;
});

TVM_REGISTER_GLOBAL("tvm.contrib.quantize.find_minmax")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* input = args[0];
  DLTensor* data_min = args[1];
  DLTensor* data_max = args[2];
  // calculate the data_min and data_max
  auto data_ptr = static_cast<float *>(input->data);
  int m = input->shape[0];
  int n = input->shape[1];
  float d_min = data_ptr[0];
  float d_max = data_ptr[0];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        d_min = std::min(data_ptr[i*n + j], d_min);
        d_max = std::max(data_ptr[i*n + j], d_max);
    }
  }
  auto out_ptr_min = static_cast<float *>(data_min->data);
  auto out_ptr_max = static_cast<float *>(data_max->data);
  *out_ptr_min =  d_min;
  *out_ptr_max =  d_max;
});

}  // namespace contrib
}  // namespace tvm
