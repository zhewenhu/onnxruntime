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

QDQSimpleChecker::QDQSimpleChecker()
    : QDQChecker{},
      dq_scale_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::SCALE_ID}},
      dq_zero_point_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::ZERO_POINT_ID}},
      q_scale_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::SCALE_ID}},
      q_zero_point_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::ZERO_POINT_ID}} {
}

bool QDQSimpleChecker::Check(const Graph& graph,
                             const Node& node,
                             const std::vector<const Node*>& dq_nodes,
                             const std::vector<const Node*>& q_nodes) const {
  if (dq_nodes.size() != 1 ||
      q_nodes.size() != 1 ||
      !optimizer_utils::CheckOutputEdges(graph, node, 1)) {
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

QDQNodePairChecker ::QDQNodePairChecker(InOutDefSlot dq_slot, InOutDefSlot q_slot)
    : dq_slot_{dq_slot},
      //dq_in_{dq_slot, DQOpName},
      dq_scale_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::SCALE_ID}},
      dq_zero_point_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::ZERO_POINT_ID}},
      q_slot_{q_slot},
      //q_out_{q_slot, QOpName},
      q_scale_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::SCALE_ID}},
      q_zero_point_is_constant_scalar_{InOutDefSlot{Direction::kInput, QDQInputIndex::ZERO_POINT_ID}} {
}

bool QDQNodePairChecker::operator()(const Graph& graph, const Node& node) const {
  const auto in_edge = std::find_if(node.InputEdgesBegin(), node.InputEdgesEnd(),
                                    [this](const Node::EdgeEnd& edge) {
                                      return (edge.GetDstArgIndex() == dq_slot_.idx);
                                    });

  if (in_edge == node.InputEdgesEnd()) {
    return false;
  }

  // current node can only have an edge to the DQ node and must not be producing a graph output
  if (!optimizer_utils::CheckOutputEdges(graph, node, 1)) {
    return false;
  }

  const auto out_edge = std::find_if(node.OutputEdgesBegin(), node.OutputEdgesEnd(),
                                     [this](const Node::EdgeEnd& edge) {
                                       return (edge.GetSrcArgIndex() == q_slot_.idx);
                                     });

  // check zp and s are scalar and const initializers and have the same values
  const auto& dq_node = in_edge->GetNode();
  const auto& q_node = out_edge->GetNode();

  if (!(dq_in_(graph, node) &&
        q_out_(graph, node) &&
        dq_scale_is_constant_scalar_(graph, dq_node) &&
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

Status MergeIntoExisting::operator()(Graph& graph, std::vector<Node*>& nodes) {
  for (const auto& src_dst : value_moves_) {
    // move inputs/output and associated edges
    ORT_RETURN_IF_ERROR(MoveInputOutput(src_dst.first, src_dst.second)(graph, nodes));
  }

  auto status = RemoveNodes(nodes_to_remove_)(graph, nodes);
  return status;
}

}  // namespace QDQ
}  // namespace onnxruntime
