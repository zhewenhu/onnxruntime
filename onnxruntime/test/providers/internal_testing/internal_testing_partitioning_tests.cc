// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/utils.h"
#include "core/session/inference_session.h"

#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/providers/internal_testing/internal_testing_execution_provider.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <queue>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {

namespace test {

// model has partitions that the initial topo sort has separate, but with the partition aware topo sort
// or merge should become one
TEST(InternalTestingEP, TestMergePartitions) {
  auto test_model = [](int get_capability_version, int& num_nodes_handled, int& num_nodes_not_handled) {
    std::cout << "Test version " << get_capability_version << "\n";

    SessionOptions so;
    // so.graph_optimization_level = TransformerLevel::Level3;

    auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

    const std::unordered_set<std::string> supported_ops{"Add"};
    const std::unordered_set<std::string> stop_ops;

    ASSERT_STATUS_OK(session->RegisterExecutionProvider(
        std::make_unique<InternalTestingExecutionProvider>(supported_ops, stop_ops, get_capability_version)));

    const ORTCHAR_T* model_path = ORT_TSTR("testdata/ep_partitioning_test_1.onnx");
    ASSERT_STATUS_OK(session->Load(model_path));
    const auto& graph = session->GetGraph();
    GraphViewer viewer{graph};

    ASSERT_STATUS_OK(session->Initialize());

    const auto& func_mgr = session->GetSessionState().GetFuncMgr();
    NodeComputeInfo* compute_func = nullptr;

    num_nodes_handled = 0;
    num_nodes_not_handled = 0;

    for (const auto& node : graph.Nodes()) {
      EXPECT_EQ(supported_ops.count(node.OpType()), size_t(0))
          << "Nodes with supported op types should have been replaced. Node with type " << node.OpType() << " was not.";
      if (node.GetExecutionProviderType() == utils::kInternalTestingExecutionProvider) {
        EXPECT_STATUS_OK(func_mgr.GetFuncs(node.Name(), compute_func));
        EXPECT_NE(compute_func, nullptr);
        ++num_nodes_handled;
      } else {
        ++num_nodes_not_handled;
      }
    }
  };

  int num_partitions_1{0}, num_partitions_3{0};
  int num_other_nodes_1{0}, num_other_nodes_3{0};

  test_model(1, num_partitions_1, num_other_nodes_1);
  test_model(3, num_partitions_3, num_other_nodes_3);

  ASSERT_EQ(num_partitions_1, 2);
  ASSERT_EQ(num_partitions_3, 1);
  ASSERT_EQ(num_other_nodes_1, 2);
  ASSERT_EQ(num_other_nodes_3, 2);
}

TEST(InternalTestingEP, TestPartitionHandling2) {
  auto test_model = [](int get_capability_version, int& num_nodes_handled, int& num_nodes_not_handled) {
    // make sure we can't save a model with compiled ops. input/output model format doesn't matter
    SessionOptions so;
    // so.graph_optimization_level = TransformerLevel::Level3;

    auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

    const std::unordered_set<std::string> supported_ops{"Add"};
    const std::unordered_set<std::string> stop_ops;

    ASSERT_STATUS_OK(session->RegisterExecutionProvider(
        std::make_unique<InternalTestingExecutionProvider>(supported_ops, stop_ops, get_capability_version)));

    const ORTCHAR_T* model_path = ORT_TSTR("testdata/ep_partitioning_test_2.onnx");
    ASSERT_STATUS_OK(session->Load(model_path));
    const auto& graph = session->GetGraph();
    GraphViewer viewer{graph};

    ASSERT_STATUS_OK(session->Initialize());

    const auto& func_mgr = session->GetSessionState().GetFuncMgr();
    NodeComputeInfo* compute_func = nullptr;

    num_nodes_handled = 0;
    num_nodes_not_handled = 0;

    for (const auto& node : graph.Nodes()) {
      EXPECT_EQ(supported_ops.count(node.OpType()), size_t(0))
          << "Nodes with supported op types should have been replaced. Node with type " << node.OpType() << " was not.";
      if (node.GetExecutionProviderType() == utils::kInternalTestingExecutionProvider) {
        EXPECT_STATUS_OK(func_mgr.GetFuncs(node.Name(), compute_func));
        EXPECT_NE(compute_func, nullptr);
        ++num_nodes_handled;
      } else {
        ++num_nodes_not_handled;
      }
    }
  };

  int num_partitions_1{0}, num_partitions_3{0};
  int num_other_nodes_1{0}, num_other_nodes_3{0};

  test_model(3, num_partitions_3, num_other_nodes_3);
  test_model(1, num_partitions_1, num_other_nodes_1);

  ASSERT_EQ(num_partitions_1, 2);
  ASSERT_EQ(num_partitions_3, 2);
  ASSERT_EQ(num_other_nodes_1, 2);
  ASSERT_EQ(num_other_nodes_3, 2);
}

static std::unordered_set<std::string> GetNnapiSupportedOps() {
  return std::unordered_set<std::string>{
      "Add",
      "Sub",
      "Mul",
      "Div",
      "QLinearAdd",
      "Pow",
      "Relu",
      "Transpose",
      "Reshape",
      "BatchNormalization",
      "GlobalAveragePool",
      "GlobalMaxPool",
      "AveragePool",
      "MaxPool",
      "QLinearAveragePool",
      "Conv",
      "QLinearConv",
      "Cast",
      "Softmax",
      "Identity",
      "Gemm",
      "MatMul",
      "QLinearMatMul",
      "Abs",
      "Exp",
      "Floor",
      "Log",
      "Sigmoid",
      "Neg",
      "Sin",
      "Sqrt",
      "Tanh",
      "QLinearSigmoid",
      "Concat",
      "Squeeze",
      "QuantizeLinear",
      "DequantizeLinear",
      "LRN",
      "Clip",
      "Resize",
      "Flatten",
      "Min",
      "Max"};
}

struct PartitionStats {
  int num_nodes_handled;
  int num_nodes_not_handled;
  int num_compiled_nodes;
};

static void TestNnapiPartitioning(const std::string& test_name,
                                  const std::string& model_uri,
                                  int get_capability_ver, bool optimize, bool debug_output,
                                  const std::unordered_set<std::string>& stop_ops,
                                  const std::vector<std::string>& additional_supported_ops,
                                  PartitionStats& stats) {
  SessionOptions so;
  so.graph_optimization_level = optimize ? TransformerLevel::Level3 : TransformerLevel::Level1;

  auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

  // we disable NCHWc in mobile scenarios as it's not relevant to ARM
  if (optimize) {
    session->FilterEnabledOptimizers({"NchwcTransformer"});
  }

  auto ops = GetNnapiSupportedOps();
  for (const auto& op_type : additional_supported_ops) {
    ops.insert(op_type);
  }

  ASSERT_STATUS_OK(session->RegisterExecutionProvider(
      std::make_unique<InternalTestingExecutionProvider>(ops, stop_ops, get_capability_ver, debug_output)));

  ASSERT_STATUS_OK(session->Load(model_uri));
  const auto& graph = session->GetGraph();
  GraphViewer viewer{graph};

  // save node count before optimization/partitioning. we lose some accuracy if optimizations replace nodes
  const auto num_nodes = graph.NumberOfNodes();

  ASSERT_STATUS_OK(session->Initialize());

  // log the unsupported ops after initializer so that anything removed by constant folding etc. isn't listed.
  std::unordered_map<std::string, int> unsupported_ops;
  std::ostringstream oss;
  std::string unsupported_op_str;

  for (const Node& node : graph.Nodes()) {
    if (node.GetExecutionProviderType() != utils::kInternalTestingExecutionProvider &&
        ops.count(node.OpType()) == 0) {
      auto entry = unsupported_ops.find(node.OpType());
      if (entry != unsupported_ops.end()) {
        ++entry->second;
      } else {
        unsupported_ops[node.OpType()] = 1;
      }
    }
  }

  if (!unsupported_ops.empty()) {
    bool add_comma = false;
    for (const auto& pair : unsupported_ops) {
      if (add_comma) {
        oss << ",";
      }

      oss << pair.first << "(" << pair.second << ")";
      add_comma = true;
    }

    unsupported_op_str = oss.str();
  }

  const auto& func_mgr = session->GetSessionState().GetFuncMgr();
  NodeComputeInfo* compute_func = nullptr;

  stats.num_nodes_not_handled = 0;
  stats.num_compiled_nodes = 0;

  // find the nodes downstream of the excluded nodes to check that they were assigned correctly
  std::unordered_set<const Node*> excluded_nodes;
  if (!stop_ops.empty()) {
    for (const auto& node : graph.Nodes()) {
      if (stop_ops.find(node.OpType()) != stop_ops.cend()) {
        excluded_nodes.insert(&node);

        // add all the downstream nodes to the excluded list
        std::queue<const Node*> nodes_to_add;
        nodes_to_add.push(&node);
        while (!nodes_to_add.empty()) {
          const Node* cur_node = nodes_to_add.front();
          nodes_to_add.pop();

          std::for_each(cur_node->OutputNodesBegin(), cur_node->OutputNodesEnd(),
                        [&nodes_to_add, &excluded_nodes](const Node& output_node) {
                          nodes_to_add.push(&output_node);
                          excluded_nodes.insert(&output_node);
                        });
        }
      }
    }
  }

  for (const auto& node : graph.Nodes()) {
    if (stop_ops.empty() || excluded_nodes.find(&node) == excluded_nodes.cend()) {
      EXPECT_EQ(ops.count(node.OpType()), size_t(0))
          << "Nodes with supported op types should have been replaced. Node with type "
          << node.OpType() << " was not.";
    } else {
      EXPECT_NE(node.GetExecutionProviderType(), utils::kInternalTestingExecutionProvider)
          << "Node is downstream from a 'stop at' node and should not have been taken. Node:"
          << node.Name();
    }

    if (node.GetExecutionProviderType() == utils::kInternalTestingExecutionProvider) {
      EXPECT_STATUS_OK(func_mgr.GetFuncs(node.Name(), compute_func));
      EXPECT_NE(compute_func, nullptr);
      ++stats.num_compiled_nodes;
    } else {
      ++stats.num_nodes_not_handled;
    }
  }

  stats.num_nodes_handled = num_nodes - stats.num_nodes_not_handled;

  auto pad_str = [](std::string const& str, size_t len = 10) {
    return (str.size() < len) ? str + std::string(len - str.size(), ' ') : str;
  };

  std::cout << pad_str(test_name, 25)
            << ": Compiled=" << stats.num_compiled_nodes
            << " Handled=" << stats.num_nodes_handled
            << " NotHandled=" << stats.num_nodes_not_handled
            << " UnsupportedOps=" << unsupported_op_str
            << "\n";
}
TEST(InternalTestingEP, TestNnapiPartitioningMlPerfModels) {
  const auto supported_ops = GetNnapiSupportedOps();

  std::vector<std::string> model_paths = {
      "D://tflite_models//mlperf_models//deeplabv3_mnv2_ade20k_float.onnx",
      "D://tflite_models//mlperf_models//deeplabv3_mnv2_ade20k_float-int8.onnx",
      "D://tflite_models//mlperf_models//mobilebert.onnx",
      "D://tflite_models//mlperf_models//mobilebert-int8.onnx",
      "D://tflite_models//mlperf_models//mobiledet.onnx",
      "D://tflite_models//mlperf_models//mobiledet-int8.onnx",
      "D://tflite_models//mlperf_models//mobilenet_edgetpu_224_1.0_float.onnx",
      "D://tflite_models//mlperf_models//mobilenet_edgetpu_224_1.0_float-int8.onnx",
      "D://tflite_models//mlperf_models//mobilenet_v1.1.0_224_opset12.onnx",
      "D://tflite_models//mlperf_models//pytorch.mobilenet_v2_float.onnx",
      "D://tflite_models//mlperf_models//pytorch.mobilenet_v2_uint8.onnx",
      "D://tflite_models//mlperf_models//resnet50_v1.onnx",
      "D://tflite_models//mlperf_models//resnet50_v1-int8.onnx",
      "D://tflite_models//mlperf_models//ssd_mobilenet_v2_300_float.onnx",
      "D://tflite_models//mlperf_models//ssd_mobilenet_v2_300_float-int8.onnx",
      "C:/Users/scmckay/Downloads/yolov4/yolov4.onnx",
  };

  // for list of mlperf models
  for (const auto model_uri : model_paths) {
    auto run_tests = [&](bool optimize) {
      std::cout << "\n\n================================\n";
      std::cout << "Model: " << model_uri;
      if (optimize) {
        std::cout << " (optimized)";
      }
      std::cout << std::endl;

      PartitionStats old_stats{}, new_stats{}, new_stop_at_nms_stats{}, new_slice_stats{}, new_slice_nms_stats{},
          extra_stats{};

      // Current
      // model, version, optimize, debug_output, stop at, extra supported, stats
      TestNnapiPartitioning("Current", model_uri, 1, optimize, false, {}, {}, old_stats);
      TestNnapiPartitioning("New", model_uri, 3, optimize, false, {}, {}, new_stats);
      TestNnapiPartitioning("New+NMS", model_uri, 3, optimize, false, {"NonMaxSuppression"}, {}, new_stop_at_nms_stats);
      TestNnapiPartitioning("New+Slice", model_uri, 3, optimize, false, {}, {"Slice"}, new_slice_stats);
      TestNnapiPartitioning("New+Slice+NMS", model_uri, 3, optimize, false, {"NonMaxSuppression"}, {"Slice"}, new_slice_nms_stats);

      // shouldn't change the nodes that are handled
      ASSERT_EQ(old_stats.num_nodes_not_handled, new_stats.num_nodes_not_handled);
    };

    run_tests(false);
    // run_tests(true);  // optimized - models have already be optimized so this isn't helpful
  }
}

TEST(MemoryAlignment, MemoryAlignmentTest) {
  std::vector<void*> to_free;

  size_t start_unaligned = 0;
  bool last_unaligned = false;

  size_t size = 0;
  size_t multiplier = 4;
  size_t per_iteration_increase = 256 * multiplier;

  int iterations = 10000;
  to_free.reserve(13);

  for (int i = 1; i < iterations; ++i) {
    size = i * per_iteration_increase;

    to_free.push_back(malloc(size));
    size_t address = reinterpret_cast<size_t>(to_free.back());

    if ((address % 64) != 0) {
      if (!last_unaligned) {
        std::cout << size << ": addr % 64 == " << address % 64 << "\n";
        start_unaligned = size;
      } else {
        // std::cout << size << "(" << address % 64 << ") ";
      }

      last_unaligned = true;
    } else {
      if (last_unaligned) {
        std::cout << "\nUnaligned from " << start_unaligned << " to " << size - per_iteration_increase << "\n";
      }
      last_unaligned = false;
    }

    if (i % 13 == 0) {
      for (auto* ptr : to_free) {
        free(ptr);
      }
      to_free.clear();
    }
  }

  if (last_unaligned) {
    std::cout << "\nUnaligned from " << start_unaligned << " to " << size << "\n";
  }

  for (auto* ptr : to_free) {
    free(ptr);
  }
}

}  // namespace test
}  // namespace onnxruntime
