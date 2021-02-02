from .base_operator import QuantOperatorBase
from ..quant_utils import QuantizedValue, QuantizedValueType


class QTranspose(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Transpose")

        # Force quantize, do not check input quantied or not

        # No assert on op_type as it is controlled by registry
        # only try to quantize when given quantization parameters for it
        data_found, output_scale_name, output_zp_name, _, _ = \
            self.quantizer._get_quantization_params(node.output[0])
        if not data_found:
            super().quantize()
            return

        nodes = []
        if node.input[0] not in self.quantizer.quantized_value_map:
            (quantized_input_names, zero_point_names, scale_names, nodes) = \
                self.quantizer.quantize_inputs(node, [0])

            # Create an entry for input quantized value
            quantized_input_value = QuantizedValue(node.input[0], quantized_input_names[0],
                                                   scale_names[0], zero_point_names[0],
                                                   QuantizedValueType.Input)
            self.quantizer.quantized_value_map[node.input[0]] = quantized_input_value

        # Create an entry for output quantized value
        quantized_input_value = self.quantizer.quantized_value_map[node.input[0]]
        quantized_output_value = QuantizedValue(node.output[0], node.output[0] + "_quantized",
                                                quantized_input_value.scale_name, quantized_input_value.zp_name,
                                                QuantizedValueType.Input)
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        node.input[0] = quantized_input_value.q_name
        node.output[0] = quantized_output_value.q_name
        nodes.append(node)

        self.quantizer.new_nodes += nodes
