// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/transformer_rules.h"

#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_rules_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
namespace QDQ {

namespace {
const Node* NodeFromSlot(const Node& node, const InOutDefSlot& slot) {
  if (slot.in_out == Direction::kInput) {
    auto iter = std::find_if(node.InputEdgesBegin(), node.InputEdgesEnd(),
                             [&slot](const Node::EdgeEnd& edge) {
                               return (edge.GetDstArgIndex() == slot.idx);
                             });

    return iter != node.InputEdgesEnd() ? &iter->GetNode() : nullptr;
  } else {
    auto iter = std::find_if(node.OutputEdgesBegin(), node.OutputEdgesEnd(),
                             [&slot](const Node::EdgeEnd& edge) {
                               return (edge.GetSrcArgIndex() == slot.idx);
                             });

    return iter != node.OutputEdgesEnd() ? &iter->GetNode() : nullptr;
  }
}

struct MoveInputOutputImpl {
  MoveInputOutputImpl(const InOutDefSlot& src_slot, const InOutDefSlot& dest_slot)
      : src_slot_{src_slot},
        dest_slot_{dest_slot},
        copy_all_{src_slot.idx == -1},
        append_{dest_slot.idx == -1} {}

  Status operator()(Graph& graph, Node& src, Node& dest) {
    ORT_RETURN_IF_ERROR(MoveNodeArg(graph, src, dest));
    MoveEdges(graph, src, dest);

    return Status::OK();
  }

  Status MoveNodeArg(Graph& graph, Node& src, Node& dest) const {
    auto& src_defs = (src_slot_.in_out == Direction::kInput)
                         ? src.MutableInputDefs()
                         : src.MutableOutputDefs();

    auto& dest_defs = (dest_slot_.in_out == Direction::kInput)
                          ? dest.MutableInputDefs()
                          : dest.MutableOutputDefs();

    auto process = [&](int src_idx) {
      ORT_ENFORCE((src_idx == -1 || src_idx < src_defs.size()) &&
                      (dest_slot_.idx == -1 || dest_slot_.idx < dest_defs.size()),
                  "Index out of range");

      if (append_) {
        dest_defs.push_back(src_defs[src_idx]);
        if (dest_slot_.in_out == Direction::kInput) {
          // TODO: If we need to support variadic inputs appending 1 each time won't work
          dest.MutableInputArgsCount().push_back(1);
        }
      } else {
        // remove any edge to the slot we're replacing
        RemoveEdge(graph, dest, dest_slot_);
        dest_defs[dest_slot_.idx] = src_defs[src_slot_.idx];
      }
    };

    if (copy_all_) {
      for (int i = 0, end = gsl::narrow<int>(src_defs.size()); i < end; ++i) {
        process(i);
      }
    } else {
      process(src_slot_.idx);
    }

    return Status::OK();
  }

  // remove edges for the src+src_slot if dest+dest_slot not provided.
  // moves edges from src+src_slot to dest node+dest_slot if provided.
  static void ProcessEdge(Graph& graph, Node& src, const InOutDefSlot& src_slot,
                          Node* dest, const InOutDefSlot* dest_slot) {
    if (src_slot.in_out == Direction::kInput) {
      // move input edge if present
      auto iter = std::find_if(src.InputEdgesBegin(), src.InputEdgesEnd(),
                               [&src_slot](const Node::EdgeEnd& edge) {
                                 return (edge.GetDstArgIndex() == src_slot.idx);
                               });

      // initializer or graph input doesn't have an edge so either zero or one edges to process
      if (iter != src.InputEdgesEnd()) {
        const Node& iter_node = iter->GetNode();
        graph.RemoveEdge(iter_node.Index(), src.Index(), iter->GetSrcArgIndex(), src_slot.idx);
        if (dest && dest_slot) {
          graph.AddEdge(iter_node.Index(), dest->Index(), iter->GetSrcArgIndex(), dest_slot->idx);
        }
      }

    } else {
      // otherwise we need to move all output edges (if any)
      auto edges = graph_utils::GraphEdge::GetNodeOutputEdges(src, src_slot.idx);
      graph_utils::GraphEdge::RemoveGraphEdges(graph, edges);
      if (dest && dest_slot) {
        for (const auto& edge : edges) {
          graph.AddEdge(dest->Index(), edge.dst_node, dest_slot->idx, edge.dst_arg_index);
        }
      }
    }
  }

  // MoveEdges to find the matching edges
  void RemoveEdge(Graph& graph, Node& node, const InOutDefSlot& slot) const {
    ProcessEdge(graph, node, slot, nullptr, nullptr);
  }

  void MoveEdges(Graph& graph, Node& src, Node& dest) const {
    ProcessEdge(graph, src, src_slot_, &dest, &dest_slot_);
  }

 private:
  InOutDefSlot src_slot_;
  InOutDefSlot dest_slot_;
  bool copy_all_;
  bool append_;
};

}  // namespace

QDQSimpleSelector::QDQSimpleSelector()
    : QDQSelector{},
      dq_scale_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::SCALE_ID}},
      dq_zero_point_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::ZERO_POINT_ID}},
      q_scale_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::SCALE_ID}},
      q_zero_point_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::ZERO_POINT_ID}} {
}

bool QDQSimpleSelector::Check(const Graph& graph,
                              const Node& node,
                              const std::vector<const Node*>& dq_nodes,
                              const std::vector<const Node*>& q_nodes) const {
  //
  if (!CheckQDQNodes(graph, node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  const Node& dq_node = *dq_nodes.front();
  const Node& q_node = *q_nodes.front();

  if (!(dq_scale_is_constant_scalar_(graph, dq_node) &&
        dq_zero_point_is_constant_scalar_(graph, dq_node) &&
        q_scale_is_constant_scalar_(graph, q_node) &&
        q_zero_point_is_constant_scalar_(graph, q_node))) {
    return false;
  }

  // check values match
  const auto& model_path = graph.ModelPath();
  const auto& dq_scale_arg = dq_node.InputDefs()[QDQInputIndex::SCALE_ID]->Name();
  const auto& dq_zp_arg = dq_node.InputDefs()[QDQInputIndex::ZERO_POINT_ID]->Name();
  const auto& q_scale_arg = q_node.InputDefs()[QDQInputIndex::SCALE_ID]->Name();
  const auto& q_zp_arg = q_node.InputDefs()[QDQInputIndex::ZERO_POINT_ID]->Name();

  Initializer dq_scale(*graph.GetConstantInitializer(dq_scale_arg, true), model_path);
  Initializer dq_zp(*graph.GetConstantInitializer(dq_zp_arg, true), model_path);
  Initializer q_scale(*graph.GetConstantInitializer(q_scale_arg, true), model_path);
  Initializer q_zp(*graph.GetConstantInitializer(q_zp_arg, true), model_path);

  return q_zp.data_type() == dq_zp.data_type() &&
         *q_zp.data<int8_t>() == *dq_zp.data<int8_t>() &&
         *q_scale.data<float>() == *dq_scale.data<float>();
}

QDQBinarySelector::QDQBinarySelector() : QDQSelector{} {}

bool QDQBinarySelector::Check(const Graph& graph,
                              const Node& node,
                              const std::vector<const Node*>& dq_nodes,
                              const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph, node, dq_nodes, q_nodes)) {
    return false;
  }

  // Currently QLinearConv only support activation type uint8_t
  int32_t dt_input_1 = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_input_2 = dq_nodes[1]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  return dt_input_1 == dt_input_2 &&
         dt_input_1 == dt_output;
}

Status MoveInputOutput::operator()(Graph& graph, std::vector<Node*>& nodes) {
  Node& src = *GetNode(source_.node_idx, nodes);
  Node& dest = *GetNode(target_.node_idx, nodes);

  return MoveInputOutputImpl(source_.in_out_slot, target_.in_out_slot)(graph, src, dest);
}

Status MergeIntoExisting::operator()(Graph& graph, std::vector<Node*>& nodes) {
  for (const auto& src_dst : value_moves_) {
    // move inputs/output and associated edges
    ORT_RETURN_IF_ERROR(MoveInputOutput(src_dst.first, src_dst.second)(graph, nodes));
  }

  auto status = RemoveNodes(nodes_to_remove_)(graph, nodes);
  return status;
}

ReplaceWithNew::ReplaceWithNew(size_t node_to_replace,
                               std::initializer_list<std::pair<NodeAndArg, InOutDefSlot>> value_moves)
    : node_to_replace_{node_to_replace},
      value_moves_{value_moves} {
}

Status ReplaceWithNew::operator()(Graph& graph, std::vector<Node*>& nodes) {
  auto& replaced_node = *GetNode(node_to_replace_, nodes);

  // create node. we'll populate the input and output defs via moves
  auto& node = graph.AddNode(replaced_node.Name(),
                             "QLinear" + replaced_node.OpType(),
                             replaced_node.Description(),
                             {},  // input defs
                             {},  // output defs
                             &replaced_node.GetAttributes(),
                             kMSDomain);

  node.SetExecutionProviderType(kCpuExecutionProvider);

  for (const auto& pair : value_moves_) {
    const NodeAndArg& src_info = pair.first;
    const InOutDefSlot& dest_slot = pair.second;

    Node& src = *GetNode(src_info.node_idx, nodes);
    MoveInputOutputImpl(src_info.in_out_slot, dest_slot)(graph, src, node);
  }

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
  const char* const name = "b33fd0fa-cd7b-4b10-ae5a-df64cabfe1f8";
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
  tensor_proto.set_raw_data(std::vector<int8_t>{0}.data(), sizeof(int8_t));

  return tensor_proto;
}();

const ONNX_NAMESPACE::TensorProto SetOptionalZeroPoint::optional_zero_point_uint8_ = []() {
  const char* const name = "b33f88f7-c464-43e3-8692-97ac832bb14a";
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name(name);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  tensor_proto.set_raw_data(std::vector<uint8_t>{0}.data(), sizeof(uint8_t));

  return tensor_proto;
}();

}  // namespace QDQ
}  // namespace onnxruntime
