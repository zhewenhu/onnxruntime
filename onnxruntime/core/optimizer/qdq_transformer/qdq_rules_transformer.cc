// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_rules_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/qdq_transformer/transformer_rules.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
namespace QDQ {
class RulesTransformerImpl {
 public:
  RulesTransformerImpl(Graph& graph, const std::unordered_map<std::string, Rules>& rules)
      : graph_{graph}, rules_{rules} {
  }

  bool IsMatch(const Node& node) const {
    // initial lookup for match on just op type
    auto op_rules = rules_.find(node.OpType());
    if (op_rules == rules_.cend()) {
      return false;
    }

    const Rules& rules = op_rules->second;

    if (node.OpType() != rules.NodeSelection.target_op_ ||
        node.Domain() != rules.NodeSelection.target_domain_) {
      return false;
    }

    // use 'for' loop for debugging. switch to std::all_of before checkin
    for (const auto& rule : rules.NodeSelection.rules) {
      if (!(*rule)(graph_, node)) {
        return false;
      }
    }

    bool redo = std::all_of(rules.NodeSelection.rules.cbegin(),
                            rules.NodeSelection.rules.cend(),
                            [this, &node](const std::unique_ptr<ConstraintChecker>& check) {
                              return (*check)(graph_, node);
                            });

    assert(redo);  // all_of should return the same result

    return true;
  }

  Status Process(const Node& /*node*/) const {
    // create new node/s

    // move edges for inputs

    // move edges for outputs

    // remove nodes
    return Status::OK();
  }

 private:
  Graph& graph_;
  const std::unordered_map<std::string, Rules>& rules_;
};
/*
  // Q/DQ contains optional input is not supported
  // non-scalar Q/DQ scale and zero point needs are not supported
  if (dq_input_defs.size() != QDQInputIndex::TOTAL_COUNT ||
      q_input_defs.size() != QDQInputIndex::TOTAL_COUNT ||
      !optimizer_utils::IsScalar(*q_input_defs[QDQInputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*q_input_defs[QDQInputIndex::ZERO_POINT_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[QDQInputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[QDQInputIndex::ZERO_POINT_ID])) {
    return false;
  }

  // if Q/DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto =
      graph_utils::GetConstantInitializer(graph, dq_input_defs[QDQInputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto =
      graph_utils::GetConstantInitializer(graph, q_input_defs[QDQInputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto =
      graph_utils::GetConstantInitializer(graph, dq_input_defs[QDQInputIndex::ZERO_POINT_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto =
      graph_utils::GetConstantInitializer(graph, q_input_defs[QDQInputIndex::ZERO_POINT_ID]->Name());
  if (nullptr == q_zp_tensor_proto ||
      nullptr == dq_zp_tensor_proto ||
      nullptr == q_scale_tensor_proto ||
      nullptr == dq_scale_tensor_proto) {
    return false;

*/

namespace {
Rules ReshapeRules() {
  std::vector<std::unique_ptr<ConstraintChecker>> rules{
      //std::make_unique<InOutCount>(Direction::kInput, QDQInputIndex::TOTAL_COUNT),
      //std::make_unique<InOutCount>(Direction::kOutput, QDQInputIndex::TOTAL_COUNT),
      //std::make_unique<ConstantInitializerChecker>(InOutDefSlot{Direction::kInput, QDQInputIndex::SCALE_ID}),
      //std::make_unique<ConstantInitializerChecker>(InOutDefSlot{Direction::kInput, QDQInputIndex::ZERO_POINT_ID}),
      //std::make_unique<ConstantInitializerChecker>(InOutDefSlot{Direction::kOutput, QDQInputIndex::SCALE_ID}),
      //std::make_unique<ConstantInitializerChecker>(InOutDefSlot{Direction::kOutput, QDQInputIndex::ZERO_POINT_ID}),
  };

  /*
  // need push_back as i
  rules.push_back(std::make_unique<InOutCount>(Direction::kInput, QDQInputIndex::TOTAL_COUNT));
  rules.push_back(std::make_unique<InOutCount>(Direction::kOutput, QDQInputIndex::TOTAL_COUNT));
  rules.push_back(std::make_unique<ConstantInitializerChecker>(InOutDefSlot{Direction::kInput, QDQInputIndex::SCALE_ID}));
  rules.push_back(std::make_unique<ConstantInitializerChecker>(InOutDefSlot{Direction::kInput, QDQInputIndex::ZERO_POINT_ID}));
  rules.push_back(std::make_unique<ConstantInitializerChecker>(InOutDefSlot{Direction::kOutput, QDQInputIndex::SCALE_ID}));
  rules.push_back(std::make_unique<ConstantInitializerChecker>(InOutDefSlot{Direction::kOutput, QDQInputIndex::ZERO_POINT_ID}));
  */
  return Rules{NodeSelectionRules{"Reshape",
                                  kOnnxDomain,
                                  std::move(rules)},
               ActionRules{}};
}
}  // namespace

RulesTransformer::RulesTransformer() noexcept
    : GraphTransformer("QDQ::RulesTransformer"),
      rules_{
          {"Reshape", ReshapeRules()}} {
}

Status RulesTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                   const logging::Logger& logger) const {
  RulesTransformerImpl impl(graph, rules_);
  GraphViewer graph_viewer(graph);  // <-- constructing a new instance is pretty costly and we should avoid in optimizers where possible

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
    if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
      if (impl.IsMatch(node)) {
        ORT_RETURN_IF_ERROR(impl.Process(node));
      }
    }
  }
  return Status::OK();
}

}  // namespace QDQ
}  // namespace onnxruntime
