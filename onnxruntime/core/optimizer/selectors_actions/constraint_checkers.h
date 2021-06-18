
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/selectors_actions/helpers.h"

namespace onnxruntime {

// helpers to check constraints
// idea is that a selector could use pre-defined checkers to handle common checks with a single implementation
// to minimize binary size, but how well that applies to other optimizers is TBD.
// A lot of the QDQ checks are across the DQ and Q nodes so
// are possibly a little harder to do with this sort of approach.
struct ConstraintChecker {
  virtual bool operator()(const Graph& graph, const Node& node) const = 0;
  virtual ~ConstraintChecker() = default;
};

// Base class for checking an input or output
// Will find the NodeArg for the given input/output and call the derived class to check it.
// If the input/output is missing, will return true if it is declared as being optional.
class InOutChecker : public ConstraintChecker {
 public:
  bool operator()(const Graph& graph, const Node& node) const override {
    const auto& node_arg = slot_.in_out == ArgType::kInput
                               ? *node.InputDefs().at(slot_.idx)
                               : *node.OutputDefs().at(slot_.idx);

    if (node_arg.Exists()) {
      return Check(graph, node_arg);
    } else {
      return optional_;
    }
  }

 protected:
  InOutChecker(InOutDefSlot slot, bool optional = false) : slot_{slot}, optional_{optional} {}
  const InOutDefSlot& Slot() const { return slot_; }

 private:
  // method for derived classes to implement
  virtual bool Check(const Graph& graph, const NodeArg& arg) const = 0;

  const InOutDefSlot slot_;
  const bool optional_;
};

class ConstantInitializerChecker : public InOutChecker {
 public:
  ConstantInitializerChecker(InOutDefSlot slot) : InOutChecker{slot} {}

 private:
  bool Check(const Graph& graph, const NodeArg& node_arg) const override {
    return graph.GetConstantInitializer(node_arg.Name(), true) != nullptr;
  }
};

class ConstantScalarChecker : public InOutChecker {
 public:
  ConstantScalarChecker(InOutDefSlot slot) : InOutChecker{slot} {}

 private:
  bool Check(const Graph& graph, const NodeArg& node_arg) const override {
    return optimizer_utils::IsScalar(node_arg) &&
           graph.GetConstantInitializer(node_arg.Name(), true) != nullptr;
  }
};

/******************
* Below checkers are currently unused. 
*/

// check that an input or output is a tensor with the specified element type
class TensorElemTypeChecker : public InOutChecker {
 public:
  TensorElemTypeChecker(InOutDefSlot slot, int32_t type) : InOutChecker{slot}, elem_type_{type} {}

 private:
  bool Check(const Graph&, const NodeArg& node_arg) const override {
    const auto* type = node_arg.TypeAsProto();
    return type != nullptr &&
           type->has_tensor_type() &&
           type->tensor_type().elem_type() == elem_type_;
  }

  const int32_t elem_type_;
};

// check the input or output count of a node.
// does not include missing optional inputs in the total.
class InOutCount : public ConstraintChecker {
 public:
  InOutCount(ArgType type, size_t count) : type_{type}, count_{count} {}

  bool operator()(const Graph&, const Node& node) const override {
    const auto& defs = type_ == ArgType::kInput ? node.InputDefs() : node.OutputDefs();
    auto num_defs = gsl::narrow_cast<size_t>(std::count_if(defs.cbegin(), defs.cend(),
                                                           [](const NodeArg* def) { return def->Exists(); }));

    return num_defs == count_;
  }

 private:
  const ArgType type_;
  const size_t count_;
};

// check if an input or output is coming from a specific OpType
class InOutOpTypeChecker : public ConstraintChecker {
 public:
  InOutOpTypeChecker(InOutDefSlot slot, const std::string& op_type)
      : slot_{slot}, op_type_{op_type} {
  }

  bool operator()(const Graph&, const Node& node) const override;

 private:
  InOutDefSlot slot_;
  const std::string op_type_;
};

}  // namespace onnxruntime
