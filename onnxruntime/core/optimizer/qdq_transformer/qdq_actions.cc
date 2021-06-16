// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/qdq_actions.h"

namespace onnxruntime {
namespace QDQ {

ReplaceWithQLinear::ReplaceWithQLinear(size_t node_to_replace,
                                       const std::string& domain,
                                       std::initializer_list<std::pair<NodeAndArg, InOutDefSlot>> value_moves)
    : node_to_replace_{node_to_replace},
      domain_{domain},
      value_moves_{value_moves} {
}

Status ReplaceWithQLinear::operator()(Graph& graph, std::vector<Node*>& nodes) {
  auto& replaced_node = *GetNode(node_to_replace_, nodes);

  // create node. we'll populate the input and output defs via moves
  auto& node = graph.AddNode(replaced_node.Name(),
                             "QLinear" + replaced_node.OpType(),
                             replaced_node.Description(),
                             {},  // input defs
                             {},  // output defs
                             &replaced_node.GetAttributes(),
                             domain_);

  node.SetExecutionProviderType(kCpuExecutionProvider);

  for (const auto& pair : value_moves_) {
    const NodeAndArg& src_info = pair.first;
    const InOutDefSlot& dest_slot = pair.second;

    // allow for optional inputs. GetNode with throw if we're missing something we expected to be there
    bool node_required = dest_slot.in_out == Direction::kOutput;
    Node* src = GetNode(src_info.node_idx, nodes, node_required);
    if (src != nullptr) {
      // TODO: Could convert value_moves_ into a collection of MoveInputOutputHelper instance so we don't do
      // that during construction. Not a big cost though.
      ORT_RETURN_IF_ERROR(MoveInputOutputHelper(src_info.in_out_slot, dest_slot).Move(graph, *src, node));
    } else {
      // optional inputs are always last in the list of things to move, so we don't need to insert a
      // nullptr in the input/output defs (TODO: Confirm)
    }
  }

  // TODO: Need to check that a DQ node providing input does not get removed if it has output edges that do not point
  // to the new node, or if it produces a graph output.
  // Could add a new 'remove nodes' variant that does this check and use it here instead of RemoveAllNodes
  RemoveAllNodes()(graph, nodes);

  return Status::OK();
}

Status SetOptionalZeroPoint::operator()(Graph& graph, std::vector<Node*>& nodes) {
  for (const size_t idx : nodes_to_update_) {
    Node& node = *GetNode(idx, nodes);
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
