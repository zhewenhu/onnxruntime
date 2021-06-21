// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_selectors.h"

#include "core/graph/graph.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

// adjust for an optional input that has an entry but does not exist
int NumExistingInputs(const Node& node) {
  const auto& input_defs = node.InputDefs();
  return gsl::narrow_cast<int>(std::count_if(input_defs.cbegin(), input_defs.cend(),
                                             [](const NodeArg* def) { return def && def->Exists(); }));
}
}  // namespace

bool QDQSelector::CheckQDQNodes(const Graph& graph, const Node& node,
                                const std::vector<const Node*>& dq_nodes,
                                const std::vector<const Node*>& q_nodes,
                                int num_dq_inputs) const {
  if (num_dq_inputs == -1) {
    num_dq_inputs = NumExistingInputs(node);
  }

  return num_dq_inputs == gsl::narrow_cast<int>(dq_nodes.size()) &&
         node.OutputDefs().size() == q_nodes.size() &&
         optimizer_utils::CheckOutputEdges(graph, node, q_nodes.size());
}

bool QDQSelector::operator()(Graph& graph, const Node& node, std::unique_ptr<NodesToOptimize>& selection) const {
  // GetDQNodes can be overridden so an op which has optional DQ inputs can insert nullptr in the correct
  // slots for those
  std::vector<const Node*> dq_nodes = graph_utils::FindParentsByType(node, QDQ::DQOpName);
  std::vector<const Node*> q_nodes = graph_utils::FindChildrenByType(node, QDQ::QOpName);

  if (!Check(graph, node, dq_nodes, q_nodes)) {
    return false;
  }

  auto get_mutable_node = [&graph](const Node* node) {
    // we use the non-const GetNode to convert the const Node* to Node*
    return graph.GetNode(node->Index());
  };

  // TODO: If this packing isn't always the case we may need to do the insertion into `selection` via a virtual
  NodesToOptimizeBuilder builder;
  builder.input_nodes.reserve(dq_nodes.size());
  builder.output_nodes.reserve(q_nodes.size());

  for (const Node* dq_node : dq_nodes) {
    builder.input_nodes.push_back(dq_node != nullptr ? get_mutable_node(dq_node) : nullptr);
  }

  builder.target_node = get_mutable_node(&node);

  for (const Node* q_node : q_nodes) {
    builder.output_nodes.push_back(get_mutable_node(q_node));
  }

  UpdateBuilder(builder);

  selection = builder.Build();

  return true;
}

QDQDropDQDNodesSelector::QDQDropDQDNodesSelector()
    : QDQSelector{},
      dq_scale_is_constant_scalar_{InOutDefSlot{ArgType::kInput, QDQ::QDQInputIndex::SCALE_ID}},
      dq_zero_point_is_constant_scalar_{InOutDefSlot{ArgType::kInput, QDQ::QDQInputIndex::ZERO_POINT_ID}},
      q_scale_is_constant_scalar_{InOutDefSlot{ArgType::kInput, QDQ::QDQInputIndex::SCALE_ID}},
      q_zero_point_is_constant_scalar_{InOutDefSlot{ArgType::kInput, QDQ::QDQInputIndex::ZERO_POINT_ID}} {
}

bool QDQDropDQDNodesSelector::Check(const Graph& graph,
                                    const Node& node,
                                    const std::vector<const Node*>& dq_nodes,
                                    const std::vector<const Node*>& q_nodes) const {
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
  const auto& dq_scale_arg = dq_node.InputDefs()[QDQ::QDQInputIndex::SCALE_ID]->Name();
  const auto& dq_zp_arg = dq_node.InputDefs()[QDQ::QDQInputIndex::ZERO_POINT_ID]->Name();
  const auto& q_scale_arg = q_node.InputDefs()[QDQ::QDQInputIndex::SCALE_ID]->Name();
  const auto& q_zp_arg = q_node.InputDefs()[QDQ::QDQInputIndex::ZERO_POINT_ID]->Name();

  // TODO: IIRC the Initializer class is pretty heavy, and given we're checking a single value here we may want
  // a cut-down version that just checks the type specific field and raw_data to read the value.
  // We can also assert that this value will not be in an external file so model_path shouldn't be necessary either
  Initializer dq_scale(*graph.GetConstantInitializer(dq_scale_arg, true), model_path);
  Initializer dq_zp(*graph.GetConstantInitializer(dq_zp_arg, true), model_path);
  Initializer q_scale(*graph.GetConstantInitializer(q_scale_arg, true), model_path);
  Initializer q_zp(*graph.GetConstantInitializer(q_zp_arg, true), model_path);

  return q_zp.data_type() == dq_zp.data_type() &&
         *q_zp.data<int8_t>() == *dq_zp.data<int8_t>() &&
         *q_scale.data<float>() == *dq_scale.data<float>();
}

bool QDQUnarySelector::Check(const Graph& graph, const Node& node,
                             const std::vector<const Node*>& dq_nodes,
                             const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph, node, dq_nodes, q_nodes, 1)) {
    return false;
  }

  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  return ((dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8 ||
           (int8_allowed_ && dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8))) &&
         ((dt_output == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8 ||
           (int8_allowed_ && dt_output == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8)));
}

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

bool QDQVariadicSelector::Check(const Graph& graph,
                                const Node& node,
                                const std::vector<const Node*>& dq_nodes,
                                const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph, node, dq_nodes, q_nodes)) {
    return false;
  }

  // All DQs' inputs and Q's output should have same data type
  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  for (size_t dq_idx = 1; dq_idx < dq_nodes.size(); dq_idx++) {
    if (dt_input != dq_nodes[dq_idx]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type()) {
      return false;
    }
  }

  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  return dt_input == dt_output;
}

void QDQVariadicSelector::UpdateBuilder(NodesToOptimizeBuilder& builder) const {
  builder.num_input_defs = 1;  // set to 1 as the first input is variadic
}

bool QDQConvSelector::Check(const Graph& graph,
                            const Node& node,
                            const std::vector<const Node*>& dq_nodes,
                            const std::vector<const Node*>& q_nodes) const {
  if (!CheckQDQNodes(graph, node, dq_nodes, q_nodes)) {
    return false;
  }

  // TODO: Can we create more generic checkers for the element type comparisons that can be re-used across the selectors
  // so that we just use that with the set of checks to make?
  //   - check type of individual input/output
  //   - check 2 types match
  //     - type selection of 1 input/output + type to compare to
  //     - or type selections of 2 inputs/outputs
  //   - check multiple types match
  // To generalize: check a collection of entries match. entries could be a type selector for an input/output or a type
  // The type selector could maybe have a cast to int for comparison against TensorProto_DataType values

  // Currently QLinearConv only support activation type uint8_t and output type uint8_t
  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  if (dt_input != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8 ||
      dt_output != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
    return false;
  }

  if (dq_nodes.size() < 3) {  // no bias
    return true;
  }

  int32_t dt_bias = dq_nodes[2]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  return dt_bias == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32;
}

void QDQConvSelector::UpdateBuilder(NodesToOptimizeBuilder& builder) const {
  builder.input_nodes.resize(3);  // add nullptr for bias if missing
}

bool QDQMatMulSelector ::Check(const Graph& graph,
                               const Node& node,
                               const std::vector<const Node*>& dq_nodes,
                               const std::vector<const Node*>& q_nodes) const {
  if (dq_nodes.size() != 2) {
    return false;
  }

  // potential match for QLinearMatMul or MatMulIntegerToFloat
  bool qlinear = !q_nodes.empty();

  if (qlinear) {
    // check for QLinearMatMul
    if (!CheckQDQNodes(graph, node, dq_nodes, q_nodes)) {
      return false;
    }

    int32_t dt_output = q_nodes[0]->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (dt_output != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
      return false;
    }
  } else {
    // MatMulIntegerToFloat
  }

  // Currently Quant MatMul only support activation type uint8_t
  int32_t dt_input = dq_nodes[0]->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  return (dt_input == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8);
}

}  // namespace onnxruntime
