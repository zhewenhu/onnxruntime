from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType

class QResize(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Resize")

        # Only after version 11, it has the optional constant_value
        # If input[0] is not quantized, do not quanitize this node
        if (self.quantizer.opset_version < 11) or (node.input[0] not in self.quantizer.quantized_value_map):
            super().quantize()
            return
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]

        # Create an entry for output quantized value
        quantized_output_value = QuantizedValue(node.output[0], node.output[0] + "_quantized",
                                                quantized_input_value.scale_name, quantized_input_value.zp_name,
                                                QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        node.input[0] = quantized_input_value.q_name
        node.output[0] = quantized_output_value.q_name
        self.quantizer.new_nodes += [node]
