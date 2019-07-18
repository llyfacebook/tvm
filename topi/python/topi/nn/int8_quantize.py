# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""int8 quantization Operators"""
from __future__ import absolute_import as _abs
import tvm
from tvm import api
from .. import tag
from ..util import simplify, get_const_int


# @tvm.target.generic_func
# def calculate_row_acc(data):
#     M, K = data.shape
#     k = tvm.reduce_axis((0, K), name='k')
#     result = tvm.compute((M, ), \
#                        lambda i: tvm.sum(data[i][k].astype("int32"), axis = k), \
#                        name='int8_row_acc', tag='int8_row_acc')
#     return [result]

@tvm.target.generic_func
def data_int8_quantize(data, zero_point, scale, is_signed, precision):
    # lambda expression
    q_min = -(1 << (precision - 1)) if is_signed else 0
    q_max = ((1 << (precision - 1)) - 1) if is_signed else (1 << precision) - 1
    target_type = "int8" if is_signed else "uint8"
    M, K = data.shape

    clamp_output = tvm.compute((M, K), \
                         lambda i, j: tvm.min(tvm.max((zero_point[0].astype("float32") \
                            + data[i][j]/scale[0]), q_min), q_max).astype(target_type), \
                         name='int8_quantize', tag='int8_quantize_clamp')

    k = tvm.reduce_axis((0, K), name='k')
    data_acc = tvm.compute((M, ), \
                         lambda i: tvm.sum(clamp_output[i][k].astype("int32"), axis = k).astype("int32"), \
                         name='int8_row_acc', tag='int8_row_acc')
    return [clamp_output, data_acc]

@tvm.target.generic_func
def data_mm_dequantize(weight, data, weight_acc, data_acc, weight_scale, activation_scale, weight_zero_point, activation_zero_point):
    # this will be merged into the FC or CONV operator
    M, K = data.shape
    N, K = weight.shape
    k = tvm.reduce_axis((0, K), name='k')
    # dense op, needs fine tuning
    quantized_mm = tvm.compute((M, N), \
                    lambda i, j: tvm.sum(data[i, k].astype("int32") * weight[j, k].astype("int32"), axis=k), \
                    name='quantized_mm', tag='quantized_mm')

    scale_multiply = tvm.compute((1, ), \
                         lambda _: (weight_scale[0] * activation_scale[0]), \
                         name = 'scale_multiply', tag='scale_multiply')

    zero_point_mulitply = tvm.compute((1, ), \
                        lambda _:(weight_zero_point[0] * activation_zero_point[0] * K).astype("float32"), \
                         name = 'zero_point_multiply', tag='zero_point_multiply')

    # need find tuning, we may decompose the complex computation.
    result = tvm.compute((M, N), \
                     lambda i, j: scale_multiply[0]*(quantized_mm[i][j].astype("float32") - \
                        data_acc[i].astype("float32")*weight_zero_point[0] - \
                        weight_acc[j].astype("float32")*activation_zero_point[0] + zero_point_mulitply[0]), \
                     name='mm_dequantized2', tag='mm_dequantized')

    return [result.astype("float32")]

@tvm.target.generic_func
def quantize_findminmax(data):
    """
    Parameters
    ----------
    Input : tvm.Tensor
        2-D with shape [M, N]

    Returns
    -------
    output : tuple including min and max
        1-D array with shape [2]
    """
    # M, N = data.shape
    # i = tvm.reduce_axis((0, M), name="i")
    # j = tvm.reduce_axis((0, N), name="j")
    # data_min = tvm.compute(
    #     (1,), lambda k: tvm.min(data[i][j], axis=[i, j]),
    #     name="FindMin", tag="quantize_min")
    # data_max = tvm.compute(
    #     (1,), lambda k: tvm.max(data[i][j], axis=[i, j]),
    #     name="FindMax", tag="quantize_max")
    # return [data_min, data_max]
    # TODO: also try call_packed(tvm.contrib.quantize.find_minmax)
    # out_bufs = []
    # data_buf = api.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    # out_bufs.append(api.decl_buffer([1], data.dtype, "max_buf", data_alignment=8))
    # out_bufs.append(api.decl_buffer([1], data.dtype, "max_buf", data_alignment=8))
    out = tvm.extern([(1,), (1,)],  [data],
                     lambda ins, outs: tvm.call_packed(
                        "tvm.contrib.quantize.find_minmax", ins[0], outs[0], outs[1]),
                      name="FindMinMax",
                      tag="FindMinMax"
                     )
    return out

@tvm.target.generic_func
def choose_quantize_params(data_min, data_max, is_signed, precision):
    q_min = -(1 << (precision - 1)) if is_signed else 0
    q_max = ((1 << (precision - 1)) - 1) if is_signed else (1 << precision) - 1
    outs = tvm.extern([(1,), (1,)],  [data_min, data_max],
                     lambda ins, outs: tvm.call_packed(
                        "tvm.contrib.quantize.choose_quantize_params", ins[0], ins[1], outs[0], outs[1], q_min, q_max),
                      name="ChooseQuantizeParams",
                      dtype=["int32", "float32"],
                      tag="ChooseQuantizeParams"
                     )
    return outs
