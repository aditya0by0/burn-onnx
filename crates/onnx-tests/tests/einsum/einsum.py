#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: einsum.onnx

import onnx
from onnx import TensorProto, helper

OPSET_VERSION = 12


def main():
    # Transpose Node
    node0 = helper.make_node(
        "Einsum", ["transpose_in"], ["transpose_out"], equation="ij->ji"
    )
    inp_transpose_in = helper.make_tensor_value_info(
        "transpose_in", TensorProto.FLOAT, [3, 4]
    )
    out_transpose_out = helper.make_tensor_value_info(
        "transpose_out", TensorProto.FLOAT, [4, 3]
    )

    # ReduceSum Node
    node1 = helper.make_node("Einsum", ["sum_in"], ["sum_out"], equation="ij->i")
    inp_sum_in = helper.make_tensor_value_info("sum_in", TensorProto.FLOAT, [3, 4])
    out_sum_out = helper.make_tensor_value_info("sum_out", TensorProto.FLOAT, [3])

    # Diagonal Node
    node2 = helper.make_node(
        "Einsum", ["batch_diagonal_in"], ["batch_diagonal_out"], equation="..ii ->...i"
    )
    inp_batch_diagonal_in = helper.make_tensor_value_info(
        "batch_diagonal_in", TensorProto.FLOAT, [3, 5, 5]
    )
    out_batch_diagonal_out = helper.make_tensor_value_info(
        "batch_diagonal_out", TensorProto.FLOAT, [3, 5]
    )

    # Inner Product Node
    node3 = helper.make_node(
        "Einsum",
        ["inner_prod_in1", "inner_prod_in2"],
        ["inner_prod_out"],
        equation="i,i",
    )
    inp_inner_prod_in1 = helper.make_tensor_value_info(
        "inner_prod_in1", TensorProto.FLOAT, [5]
    )
    inp_inner_prod_in2 = helper.make_tensor_value_info(
        "inner_prod_in2", TensorProto.FLOAT, [5]
    )
    out_inner_prod_out = helper.make_tensor_value_info(
        "inner_prod_out", TensorProto.FLOAT, []
    )

    # Batch Matrix Multiplication Node
    node4 = helper.make_node(
        "Einsum",
        ["batch_matmul_in1", "batch_matmul_in2"],
        ["batch_matmul_out"],
        equation="bij, bjk -> bik",
    )
    inp_batch_matmul_in1 = helper.make_tensor_value_info(
        "batch_matmul_in1", TensorProto.FLOAT, [5, 2, 3]
    )
    inp_batch_matmul_in2 = helper.make_tensor_value_info(
        "batch_matmul_in2", TensorProto.FLOAT, [5, 3, 4]
    )
    out_batch_matmul_out = helper.make_tensor_value_info(
        "batch_matmul_out", TensorProto.FLOAT, [5, 2, 4]
    )

    # Scalar Output Node
    node5 = helper.make_node(
        "Einsum",
        ["scalar_in"],
        ["scalar_out"],
        equation="->",
    )
    inp_scalar_in = helper.make_tensor_value_info("scalar_in", TensorProto.FLOAT, [])
    out_scalar_out = helper.make_tensor_value_info("scalar_out", TensorProto.FLOAT, [])

    graph = helper.make_graph(
        [node0, node1, node2, node3, node4, node5],
        "torch_jit",
        [
            inp_transpose_in,
            inp_sum_in,
            inp_batch_diagonal_in,
            inp_inner_prod_in1,
            inp_inner_prod_in2,
            inp_batch_matmul_in1,
            inp_batch_matmul_in2,
            inp_scalar_in,
        ],
        [
            out_transpose_out,
            out_sum_out,
            out_batch_diagonal_out,
            out_inner_prod_out,
            out_batch_matmul_out,
            out_scalar_out,
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )

    onnx.save(model, "einsum.onnx")
    print("Finished exporting model to einsum.onnx")


if __name__ == "__main__":
    main()
