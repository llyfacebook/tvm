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
 *  Copyright (c) 2018 by Contributors
 * \file nn.cc
 * \brief Property def of nn operators.
 */

#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/image.h>
#include <topi/nn.h>
#include <topi/nn/bias_add.h>
#include <topi/nn/softmax.h>
#include <topi/nn/flatten.h>
#include <vector>
#include "../type_relations.h"
#include "../../pass/alter_op_layout.h"
#include "../op_common.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(QuantizedParamsAttrs);

Expr MakeDataInt8Quantization(Expr data, Expr zero_point, Expr scale, bool is_signed, int precision) {
  static const Op& op = Op::Get("nn.contrib_quantize_data_int8_quantize");
  auto attrs = make_node<QuantizedParamsAttrs>();
  attrs->precision = precision;
  attrs->is_signed = is_signed;
  return CallNode::make(op, {data, zero_point, scale}, Attrs(attrs), {});
}

bool DataInt8QuantizationRel(const Array<Type>& types,
                             int num_inputs,
                             const Attrs& attrs,
                             const TypeReporter& reporter) {
  // todo: add axis to decide which dim to do the accumulation
  CHECK_EQ(types.size(), 4);
  const QuantizedParamsAttrs* param = attrs.as<QuantizedParamsAttrs>();
  const auto* data = types[0].as<TensorTypeNode>();
  // unchnaged shape
  Array<tvm::Expr> oshape = data->shape;
  Array<tvm::Expr> acc_oshape = {oshape[0]};

  DataType out_dtype;
  if(param->is_signed) {
    out_dtype = Int(param->precision);
  } else {
    out_dtype = UInt(param->precision);
  }
  std::vector<Type> fields;
  fields.push_back(TensorTypeNode::make(oshape, out_dtype));
  fields.push_back(TensorTypeNode::make(acc_oshape, Int(32)));
  reporter->Assign(types[3], TupleTypeNode::make(Array<Type>(fields)));
  //  reporter->Assign(types[3], TensorTypeNode::make(oshape, out_dtype));
  return true;
}


TVM_REGISTER_API("relay.op.nn._make.contrib_quantize_data_int8_quantize")
.set_body_typed(MakeDataInt8Quantization);


RELAY_REGISTER_OP("nn.contrib_quantize_data_int8_quantize")
.describe(R"code(dynamic quantization of weight or activation.
- **weight**: (channels, in_channels)
)code" TVM_ADD_FILELINE)
.set_num_inputs(3)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("zero_point", "Tensor", "The zero_point parameter for quantization")
.add_argument("scale", "Tensor", "the scale parameter for quantization")
.set_attrs_type_key("relay.attrs.QuantizedParamsAttrs")
.set_support_level(10)
.add_type_rel("DataInt8Quantization", DataInt8QuantizationRel);


Expr MakeFindMinMax(Expr data) {
  static const Op& op = Op::Get("nn.contrib_quantize_findminmax");
  return CallNode::make(op, {data}, Attrs(), {});
}

bool FindMinMaxRel(const Array<Type>& types,
                   int num_inputs,
                   const Attrs& attrs,
                   const TypeReporter& reporter) {
    CHECK_EQ(types.size(), 2);
    const auto* data = types[0].as<TensorTypeNode>();
    std::vector<IndexExpr> oshape({1});
    std::vector<Type> fields;
    fields.push_back(TensorTypeNode::make(oshape, data->dtype));
    fields.push_back(TensorTypeNode::make(oshape, data->dtype));
    reporter->Assign(types[1], TupleTypeNode::make(Array<Type>(fields)));
    return true;
}

TVM_REGISTER_API("relay.op.nn._make.contrib_quantize_findminmax")
.set_body_typed(MakeFindMinMax);

RELAY_REGISTER_OP("nn.contrib_quantize_findminmax")
.describe(R"code(find min and max of the input data.
- **data**: (M, N)
)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input data tensor.")
.set_support_level(5)
.add_type_rel("FindMinMax", FindMinMaxRel);


Expr MakeDataMMDequantize(Expr weight,
                          Expr data,
                          Expr weight_acc,
                          Expr data_acc,
                          Expr weight_scale,
                          Expr activation_scale,
                          Expr weight_zero_point,
                          Expr activation_zero_point) {
  static const Op& op = Op::Get("nn.contrib_quantize_data_mm_dequantize");
  return CallNode::make(op, {weight, data, weight_acc, data_acc, weight_scale, activation_scale, weight_zero_point, activation_zero_point}, Attrs(), {});
}

bool DataMMDequantizeRel(const Array<Type>& types,
                             int num_inputs,
                             const Attrs& attrs,
                             const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 9);
  const auto* weight = types[0].as<TensorTypeNode>();
  const auto* data = types[1].as<TensorTypeNode>();
  // TODO: check the acc shape
  // Assume acc32 input
  Array<tvm::Expr> wshape = weight->shape;
  Array<tvm::Expr> oshape = data->shape;
  oshape.Set((oshape.size() - 1), wshape[0]);
  reporter->Assign(types[8], TensorTypeNode::make(oshape, Float(32)));
  return true;
}

TVM_REGISTER_API("relay.op.nn._make.contrib_quantize_data_mm_dequantize")
.set_body_typed(MakeDataMMDequantize);

RELAY_REGISTER_OP("nn.contrib_quantize_data_mm_dequantize")
.describe(R"code(multiply the weight and data, then dequantize the data into floating point.
- **data**: (M, N)
)code" TVM_ADD_FILELINE)
.set_num_inputs(8)
.add_argument("data", "Tensor", "The input data tensor.")
.add_argument("weight", "Tensor", "The input weight tensor.")
.add_argument("data_acc", "Tensor", "The accumulation of each row")
.add_argument("weight_acc", "Tensor", "The accumulation of each column")
.add_argument("weight_scale", "Tensor", "The weight scale")
.add_argument("activation_scale", "Tensor", "The activation scale")
.add_argument("weight_zero_point", "Tensor", "The weight zero point")
.add_argument("activation_zero_point", "Tensor", "The activation zero_point")
.set_support_level(10)
.add_type_rel("DataMMDequantize", DataMMDequantizeRel);


Expr MakeChooseQuantizeParams(Expr data_min, Expr data_max, bool is_signed, int precision) {
  auto attrs = make_node<QuantizedParamsAttrs>();
  attrs->precision = precision;
  attrs->is_signed = is_signed;
  static const Op& op = Op::Get("nn.contrib_choose_quantize_params");
  return CallNode::make(op, {data_min, data_max}, Attrs(attrs), {});
}

bool ChooseQuantizeParamsRel(const Array<Type>& types,
                             int num_inputs,
                             const Attrs& attrs,
                             const TypeReporter& reporter) {
    CHECK_EQ(types.size(), 3);
    const auto* data = types[0].as<TensorTypeNode>();
    std::vector<IndexExpr> oshape({1});
    std::vector<Type> fields;
    fields.push_back(TensorTypeNode::make(oshape, Int(32)));
    fields.push_back(TensorTypeNode::make(oshape, data->dtype));
    reporter->Assign(types[2], TupleTypeNode::make(Array<Type>(fields)));
    return true;
}

TVM_REGISTER_API("relay.op.nn._make.contrib_choose_quantize_params")
.set_body_typed(MakeChooseQuantizeParams);

RELAY_REGISTER_OP("nn.contrib_choose_quantize_params")
.describe(R"code(calculate the zero_point and scale.
)code" TVM_ADD_FILELINE)
.set_num_inputs(2)
.set_attrs_type_key("relay.attrs.QuantizedParamsAttrs")
.add_argument("data_min", "Tensor", "The min of input data.")
.add_argument("data_max", "Tensor", "The max of input data.")
.set_support_level(4)
.add_type_rel("ChooseQuantizeParams", ChooseQuantizeParamsRel);


// Expr MakeCalculateRowAcc(Expr data) {
//   static const Op& op = Op::Get("nn.contrib_calculate_row_acc");
//   return CallNode::make(op, {data}, Attrs(), {});
// }
//
// bool CalculateRowAccRel(const Array<Type>& types,
//                              int num_inputs,
//                              const Attrs& attrs,
//                              const TypeReporter& reporter) {
//     CHECK_EQ(types.size(), 2);
//     const auto* data = types[0].as<TensorTypeNode>();
//     Array<tvm::Expr> acc_oshape = {data->shape[0]};
//     reporter->Assign(types[1], TensorTypeNode::make(acc_oshape, Int(32)));
//     return true;
// }
//
// TVM_REGISTER_API("relay.op.nn._make.contrib_calculate_row_acc")
// .set_body_typed(MakeCalculateRowAcc);
//
// RELAY_REGISTER_OP("nn.contrib_calculate_row_acc")
// .describe(R"code(calculate the row accumulation.
// )code" TVM_ADD_FILELINE)
// .set_num_inputs(1)
// .add_argument("data", "Tensor", "The input data.")
// .set_support_level(4)
// .add_type_rel("CalculateRowAcc", CalculateRowAccRel);

}  // namespace relay
}  // namespace tvm
