#include "core/optimizer/selectors_actions/constraint_checkers.h"

namespace onnxruntime {
namespace {
const Node* NodeFromSlot(const Node& node, const InOutDefSlot& slot) {
  if (slot.in_out == ArgType::kInput) {
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
}  // namespace

bool InOutOpTypeChecker::operator()(const Graph&, const Node& node) const {
  const Node* in_out_node = NodeFromSlot(node, slot_);
  return in_out_node && in_out_node->OpType() == op_type_;
}

}  // namespace onnxruntime
