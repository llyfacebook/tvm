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
""" Support dynamic quantization related operator test cases.
"""
import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import ctx_list


def test_quantize_findminmax():
    shape = (100, 200)
    def verify_findminmax(shape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.nn.contrib_quantize_findminmax(x)
        func = relay.Function([x], z.astuple())
        x_data = np.random.uniform(size=shape).astype("float32")

        ref_min = np.amin(x_data)
        ref_max = np.amax(x_data)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res[0].asnumpy(), ref_min.astype("float32"), rtol=1e-5)
                tvm.testing.assert_allclose(op_res[1].asnumpy(), ref_max.astype("float32"), rtol=1e-5)

    verify_findminmax(shape)

def test_choose_quantize_params():
    def verify_choose_quantize_params():
        data_min = relay.var("d_min", relay.TensorType((1,), "float32"))
        data_max = relay.var("d_max", relay.TensorType((1,), "float32"))
        q_params = relay.nn.contrib_choose_quantize_params(data_min, data_max, True, 8)
        func = relay.Function([data_min, data_max], q_params.astuple())
        x_min = np.arange(1).astype("float32")
        x_min[0] = -0.0456
        x_max = np.arange(1).astype("float32")
        x_max[0] = 0.4556
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_min, x_max)
                # print(op_res)
                # add checking logic

    verify_choose_quantize_params()

def test_data_int8_quantize():
    shape = (100, 200)
    def verify_int8_quantize(shape):
        data = relay.var("data", relay.TensorType(shape, "float32"))
        zero_point = relay.var("zero_point", relay.TensorType((1,), "int32"))
        scale = relay.var("scale", relay.TensorType((1,), "float32"))
        qdata = relay.nn.contrib_quantize_data_int8_quantize(data, zero_point, scale, 1, 8)
        func = relay.Function([data, zero_point, scale], qdata.astuple())
        in_data = np.random.uniform(size=shape).astype("float32")
        in_zero_point = np.arange(1).astype("int32")
        in_zero_point[0] = -12
        in_scale = np.arange(1).astype("float32")
        in_scale[0] = 0.0035

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(in_data, in_zero_point, in_scale)
                # todo: check the correctness

    verify_int8_quantize(shape)


# def test_data_row_acc():
#     shape = (100, 200)
#     def verify_data_row_acc(shape):
#         data = relay.var("data", relay.TensorType(shape, "int8"))
#         data_acc = relay.nn.contrib_calculate_row_acc(data)
#         func = relay.Function([data], data_acc)
#         in_data = np.random.randint(10, size=shape).astype("int8")
#         for target, ctx in ctx_list():
#             for kind in ["graph", "debug"]:
#                 intrp = relay.create_executor(kind, ctx=ctx, target=target)
#                 op_res = intrp.evaluate(func)(in_data)
#                 # todo: check the correctness
#
#     verify_data_row_acc(shape)


def test_data_mm_int8_dequantize():
    data_shape = (100, 200)
    weight_shape = (200, 200)
    def verify_int8_dequantize(data_shape, weight_shape):
        data = relay.var("data", relay.TensorType(data_shape, "int8"))
        weight = relay.var("weight", relay.TensorType(weight_shape, "int8"))
        data_acc = relay.var("data_acc", relay.TensorType((100, ), "int32"))
        weight_acc = relay.var("weight_acc", relay.TensorType((200, ), "int32"))
        w_zero_point = relay.var("w_zero_point", relay.TensorType((1,), "int32"))
        w_scale = relay.var("w_scale", relay.TensorType((1,), "float32"))
        d_zero_point = relay.var("d_zero_point", relay.TensorType((1,), "int32"))
        d_scale = relay.var("d_scale", relay.TensorType((1,), "float32"))

        de_data = relay.nn.contrib_quantize_data_mm_dequantize(weight, data, weight_acc, data_acc, w_zero_point, d_zero_point, w_scale, d_scale)
        func = relay.Function([weight, data, weight_acc, data_acc, w_zero_point, d_zero_point, w_scale, d_scale], de_data)
        in_weight = np.random.randint(5, size=weight_shape).astype("int8")
        in_data = np.random.randint(10, size=data_shape).astype("int8")
        in_weight_acc = np.random.randint(5, size=(200, )).astype("int32")
        in_data_acc = np.random.randint(10, size=(100, )).astype("int32")
        in_w_zero_point = np.arange(1).astype("int32")
        in_w_zero_point[0] = -12
        in_w_scale = np.arange(1).astype("float32")
        in_w_scale[0] = 0.0035

        in_d_zero_point = np.arange(1).astype("int32")
        in_d_zero_point[0] = -12
        in_d_scale = np.arange(1).astype("float32")
        in_d_scale[0] = 0.0035

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(in_weight, in_data, in_weight_acc, in_data_acc, in_w_zero_point, in_d_zero_point, \
                                                in_w_scale, in_d_scale)
                # todo: check the correctness
    verify_int8_dequantize(data_shape, weight_shape)


def test_quantized_e2e():
    data_shape = (300, 500)
    weight_shape = (400, 500)
    def verify_quantize_e2e(data_shape, weight_shape):
        data = relay.var("data", relay.TensorType(data_shape, "float32"))
        weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))

        x_data = np.random.uniform(low=0, high=1, size=data_shape).astype("float32")
        x_weight = np.random.uniform(low=0, high=1, size=weight_shape).astype("float32")
        x_ref = np.dot(x_data, x_weight.T)

        data_min, data_max = relay.nn.contrib_quantize_findminmax(data)
        weight_min, weight_max = relay.nn.contrib_quantize_findminmax(weight)
        data_zero_point, data_scale = relay.nn.contrib_choose_quantize_params(data_min, data_max, False, 8)
        weight_zero_point, weight_scale = relay.nn.contrib_choose_quantize_params(weight_min, weight_max, True, 8)
        data_q, data_acc = relay.nn.contrib_quantize_data_int8_quantize(data, data_zero_point, data_scale, False, 8)
        weight_q, weight_acc = relay.nn.contrib_quantize_data_int8_quantize(weight, weight_zero_point, weight_scale, True, 8)
        result = relay.nn.contrib_quantize_data_mm_dequantize(weight_q, data_q, weight_acc, data_acc, weight_scale, data_scale, weight_zero_point, data_zero_point)

        func = relay.Function([data, weight], result)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, x_weight)
                tvm.testing.assert_allclose(op_res.asnumpy(), x_ref, rtol=1e-2)

    verify_quantize_e2e(data_shape, weight_shape)


if __name__ == "__main__":
    test_quantize_findminmax()
    test_choose_quantize_params()
    # test_data_row_acc()
    test_data_int8_quantize()
    test_data_mm_int8_dequantize()
    test_quantized_e2e()
