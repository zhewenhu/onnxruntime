// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/qdq_actions.h"

namespace onnxruntime {
namespace QDQ {

ReplaceWithQLinear::ReplaceWithQLinear(const std::string& domain,
                                       std::vector<NodeAndMoveInfo>&& value_moves)
    : domain_{domain},
      value_moves_{std ::move(value_moves)} {
}

Status ReplaceWithQLinear::operator()(Graph& graph, const NodesToOptimize& selected_nodes) const {
  auto& target = *selected_nodes.Target();

  // create node. we'll populate the input and output defs via moves
  auto& replacement = graph.AddNode(target.Name(),
                                    "QLinear" + target.OpType(),
                                    target.Description(),
                                    {},  // input defs
                                    {},  // output defs
                                    &target.GetAttributes(),
                                    domain_);

  replacement.SetExecutionProviderType(kCpuExecutionProvider);

  for (const auto& move : value_moves_) {
    // get the nodes to copy from. allow for an optional input node (e.g. bias input to Conv)
    auto src_nodes = selected_nodes.GetNodesAtLocation(move.src_node, /*required*/ false);

    ORT_ENFORCE(src_nodes.size() == 1 || move.value_move_info.append == true,
                "Move of variadic values requires 'append' to be specific.");

    for (Node* src : src_nodes) {
      if (src != nullptr) {
        ORT_RETURN_IF_ERROR(MoveInputOutputHelper::Move(graph, *src, replacement, move.value_move_info));
      }
    }
  }

  auto status = RemoveNodes()(graph, selected_nodes);

  return status;
}

Status SetOptionalZeroPoint::operator()(Graph& graph, const NodesToOptimize& selected_nodes) const {
  for (const NodesToOptimize::NodeLocation& location : nodes_to_update_) {
    // generally a single node, but if we're applying this to a variadic input or output there could be multiple
    std::vector<Node*> nodes = selected_nodes.GetNodesAtLocation(location);
    for (Node* node_ptr : nodes) {
      Node& node = *node_ptr;  // GetNodesAtLocation requires the nodes to exist so we know this is not null

      std::vector<NodeArg*>& input_defs = node.MutableInputDefs();
      bool has_zp_input = input_defs.size() - 1 == QDQInputIndex::ZERO_POINT_ID;
      if (has_zp_input && input_defs[QDQInputIndex::ZERO_POINT_ID]->Exists()) {
        continue;  // zero point was set. No need to fill.
      }

      bool is_default_zp_signed = false;
      if (node.OpType() == DQOpName) {
        auto input_type = input_defs[0]->TypeAsProto()->tensor_type().elem_type();
        is_default_zp_signed = ONNX_NAMESPACE::TensorProto_DataType_INT8 == input_type;
      }

      const ONNX_NAMESPACE::TensorProto& zp_tensor_proto = is_default_zp_signed
                                                               ? optional_zero_point_int8_
                                                               : optional_zero_point_uint8_;

      const ONNX_NAMESPACE::TensorProto* dummy_zp_tensor_proto;
      if (!graph.GetInitializedTensor(zp_tensor_proto.name(), dummy_zp_tensor_proto)) {
        graph.AddInitializedTensor(zp_tensor_proto);
      }

      auto& node_arg = graph.GetOrCreateNodeArg(zp_tensor_proto.name(), nullptr);
      if (has_zp_input) {
        input_defs.push_back(&node_arg);
      } else {
        input_defs[QDQInputIndex::ZERO_POINT_ID] = &node_arg;
      }
    }
  }

  return Status::OK();
}

const ONNX_NAMESPACE::TensorProto SetOptionalZeroPoint::optional_zero_point_int8_ = []() {
  const char* const name = "b33fd0fa-cd7b-4b10-ae5a-df64cabfe1f8";  // guid as arbitrary name to provide a unique value
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
  tensor_proto.set_raw_data(std::vector<int8_t>{0}.data(), sizeof(int8_t));

  return tensor_proto;
}();

const ONNX_NAMESPACE::TensorProto SetOptionalZeroPoint::optional_zero_point_uint8_ = []() {
  const char* const name = "b33f88f7-c464-43e3-8692-97ac832bb14a";  // guid as arbitrary name to provide a unique value
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  tensor_proto.set_raw_data(std::vector<uint8_t>{0}.data(), sizeof(uint8_t));

  return tensor_proto;
}();

}  // namespace QDQ
}  // namespace onnxruntime
