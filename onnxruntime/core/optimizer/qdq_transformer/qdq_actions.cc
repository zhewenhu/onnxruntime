// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/qdq_actions.h"

namespace onnxruntime {
namespace QDQ {

Status MatMulAction ::operator()(Graph& graph, const NodesToOptimize& selected_nodes) const {
  using NTO = NodesToOptimize;

  // if the output is empty there were no Q nodes selected, so replace with MatMulIntegerToFloat
  // otherwise replace with QLinearMatMul
  bool matmul_integer_to_float = selected_nodes.num_outputs == 0;
  if (matmul_integer_to_float) {
    NTO::NodeLocation dq1{NTO::NodeType::kInput, 0};
    NTO::NodeLocation dq2{NTO::NodeType::kInput, 1};
    NTO::NodeLocation target{NTO::NodeType::kTarget, 0};

    std::vector<NodeAndMoveInfo> moves{
        MoveAndAppend(dq1, ArgType::kInput, 0, ArgType::kInput),
        MoveAndAppend(dq2, ArgType::kInput, 0, ArgType::kInput),
        MoveAndAppend(dq1, ArgType::kInput, 1, ArgType::kInput),
        MoveAndAppend(dq2, ArgType::kInput, 1, ArgType::kInput),
        MoveAndAppend(dq1, ArgType::kInput, 2, ArgType::kInput),
        MoveAndAppend(dq2, ArgType::kInput, 2, ArgType::kInput),
        MoveAll(target, ArgType::kOutput)};

    return ReplaceWithNew(kMSDomain, "MatMulIntegerToFloat", std::move(moves))(graph, selected_nodes);
  } else {
    NTO::NodeLocation dq1{NTO::NodeType::kInput, 0};
    NTO::NodeLocation dq2{NTO::NodeType::kInput, 1};
    NTO::NodeLocation q{NTO::NodeType::kOutput, 0};

    std::vector<NodeAndMoveInfo> moves{
        MoveAll(dq1, ArgType::kInput),                           // append all inputs from dq to new node
        MoveAll(dq2, ArgType::kInput),                           // append all inputs from dq to new node
        MoveAndAppend(q, ArgType::kInput, 1, ArgType::kInput),   // append scale (input 1) from q
        MoveAndAppend(q, ArgType::kInput, 2, ArgType ::kInput),  // append zp (input 2) from q
        MoveAll(q, ArgType::kOutput)};

    return ReplaceWithQLinear(kOnnxDomain, std::move(moves))(graph, selected_nodes);
  }
}

Status SetOptionalZeroPoint::operator()(Graph& graph, const NodesToOptimize& selected_nodes) const {
  std::vector<Node*> nodes = selected_nodes.AllNodes();
  for (Node* node_ptr : nodes) {
    if (node_ptr == nullptr) {
      continue;
    }

    Node& node = *node_ptr;

    bool is_dq = node.OpType() == DQOpName;
    bool is_q = node.OpType() == QOpName;
    if (!is_dq && !is_q) {
      continue;
    }

    std::vector<NodeArg*>& input_defs = node.MutableInputDefs();
    bool has_zp_input = input_defs.size() == 3;
    if (has_zp_input && input_defs[QDQInputIndex::ZERO_POINT_ID]->Exists()) {
      continue;  // zero point was set. No need to fill.
    }

    bool is_default_zp_signed = false;
    if (is_dq) {
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
