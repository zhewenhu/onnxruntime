import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

# Create graph with Add and Sub nodes that can be used to test partitioning when one of the operators
# can run using the test EP and the other cannot.
# As the operators take 2 inputs and produce one output we can easily create edges to test different scenarios

def merge_2_branches():
    # Graph where the 3 'add' nodes should be able to be run as a single partition.
    # Naively you get 2 partitions if the topo sort is left to right, top down and by looking at edges as the bottom
    # right 'add' a4 is last in the topo sort so is separated from the first partition (a1 + a3 via edge, + a2 if next
    # in topo sort and added to the group).
    # But if you move the second 'sub' node up so it runs immediately after the first one, all the 'add' nodes can
    # be merged as all their inputs are available at that point
    #
    #        input0, input1
    #         /  \  \
    #        s1  a1  \
    #       / \    |  \
    #     a2   s2  |   \
    #           \  |    \
    #             a3     a4
    graph = helper.make_graph(
        nodes =
        [
            helper.make_node("Sub", ['input0', 'input1'], ["1"], "S1"),
            helper.make_node("Add", ['input0', 'input1'], ['2'], "A1"),
            helper.make_node("Add", ['1', 'input1'], ['3_out'], "A2"),
            helper.make_node("Sub", ['1', 'input1'], ['4'], "S2"),
            helper.make_node("Add", ['2', '4'], ['5_out'], "A3"),
            helper.make_node("Add", ['input0', 'input1'], ['6_out'], "A4"),
        ],
        name = "graph",
        inputs =
        [
            helper.make_tensor_value_info('input0', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('input1', TensorProto.INT64, [1]),
        ],
        outputs =
        [
            helper.make_tensor_value_info('3_out', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('5_out', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('6_out', TensorProto.INT64, [1]),
        ],
        initializer = []
    )


    model = helper.make_model(graph)
    onnx.save(model, r'D:\src\github\ort.vs19\onnxruntime\test\testdata\merge_partitions.onnx')

merge_2_branches()