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

//// TODO: Refactor cut-and-paste of this so it can be re-used
//static void CreateSession(const SessionOptions& so, std::unique_ptr<InferenceSessionWrapper>& session,
//                          const ORTCHAR_T* model_path = ORT_TSTR("testdata/mnist.onnx"),  // arbitrary test model
//                          bool enable_custom_ep = true,
//                          const std::unordered_set<std::string>* override_supported_ops = nullptr) {
//  session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());
//
//  // set supported ops to ops that are ideally found consecutively in the model.
//  // we can say the EP potentially handles them all, but can also test removing handling of one or more ops
//  // at runtime to simulate a lower spec device where not all ops can be handled. this allows us to test
//  // that we can revert ops back to the CPU implementation successfully
//  const std::unordered_set<std::string> default_supported_ops{"Conv", "Add", "Relu", "MaxPool"};
//  const std::unordered_set<std::string>* supported_ops = override_supported_ops ? override_supported_ops
//                                                                                : &default_supported_ops;
//
//  if (enable_custom_ep) {
//    ASSERT_STATUS_OK(session->RegisterExecutionProvider(
//        std::make_unique<InternalTestingExecutionProvider>(*supported_ops)));
//  }
//
//  ASSERT_STATUS_OK(session->Load(model_path));
//  ASSERT_STATUS_OK(session->Initialize());
//}

// model has partitions that the initial topo sort has separate, but with the partition aware topo sort
// or merge should become one
// TODO: See if the merge is needed
TEST(InternalTestingEP, TestMergePartitions) {
  auto test_model = [](int get_capability_version, int& num_nodes_handled, int& num_nodes_not_handled) {
    std::cout << "Test version " << get_capability_version << "\n";

    // make sure we can't save a model with compiled ops. input/output model format doesn't matter
    SessionOptions so;
    // so.graph_optimization_level = TransformerLevel::Level3;

    auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

    const std::unordered_set<std::string> supported_ops{"Add"};
    ASSERT_STATUS_OK(session->RegisterExecutionProvider(
        std::make_unique<InternalTestingExecutionProvider>(supported_ops, get_capability_version)));

    const ORTCHAR_T* model_path = ORT_TSTR("testdata/ep_partitioning_test_1.onnx");
    ASSERT_STATUS_OK(session->Load(model_path));
    const auto& graph = session->GetGraph();
    GraphViewer viewer{graph};

    //InternalTestingExecutionProvider ep{supported_ops};
    //auto result = ep.GetCapability(viewer, {});
    //auto result2 = ep.GetCapability2(viewer, {});

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

  int num_partitions_1{0}, num_partitions_2{0}, num_partitions_3{0};
  int num_other_nodes_1{0}, num_other_nodes_2{0}, num_other_nodes_3{0};

  test_model(1, num_partitions_1, num_other_nodes_1);
  test_model(2, num_partitions_2, num_other_nodes_2);
  test_model(3, num_partitions_3, num_other_nodes_3);

  ASSERT_EQ(num_partitions_1, 2);
  ASSERT_EQ(num_partitions_2, 1);
  ASSERT_EQ(num_partitions_3, 1);
  ASSERT_EQ(num_other_nodes_1, 2);
  ASSERT_EQ(num_other_nodes_2, 2);
  ASSERT_EQ(num_other_nodes_3, 2);
}

TEST(InternalTestingEP, TestPartitionHandling2) {
  auto test_model = [](int get_capability_version, int& num_nodes_handled, int& num_nodes_not_handled) {
    // make sure we can't save a model with compiled ops. input/output model format doesn't matter
    SessionOptions so;
    // so.graph_optimization_level = TransformerLevel::Level3;

    auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

    const std::unordered_set<std::string> supported_ops{"Add"};
    ASSERT_STATUS_OK(session->RegisterExecutionProvider(
        std::make_unique<InternalTestingExecutionProvider>(supported_ops, get_capability_version)));

    const ORTCHAR_T* model_path = ORT_TSTR("testdata/ep_partitioning_test_2.onnx");
    ASSERT_STATUS_OK(session->Load(model_path));
    const auto& graph = session->GetGraph();
    GraphViewer viewer{graph};

    //InternalTestingExecutionProvider ep{supported_ops};
    //auto result = ep.GetCapability(viewer, {});
    //auto result2 = ep.GetCapability2(viewer, {});
    //auto result3 = ep.GetCapability3(viewer, {});

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

  int num_partitions_1{0}, /*num_partitions_2{0},*/ num_partitions_3{0};
  int num_other_nodes_1{0}, /*num_other_nodes_2{0},*/ num_other_nodes_3{0};

  test_model(3, num_partitions_3, num_other_nodes_3);
  // test_model(2, num_partitions_2, num_other_nodes_2);
  test_model(1, num_partitions_1, num_other_nodes_1);

  ASSERT_EQ(num_partitions_1, 2);
  //ASSERT_EQ(num_partitions_2, 2);
  ASSERT_EQ(num_partitions_3, 2);
  ASSERT_EQ(num_other_nodes_1, 2);
  //ASSERT_EQ(num_other_nodes_2, 2);
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
      "Max",
      "Slice",
  };
}

struct PartitionStats {
  int num_nodes_handled;
  int num_nodes_not_handled;
  int num_compiled_nodes;
};

TEST(InternalTestingEP, TestNnapiPartitioningMlPerfModels) {
  const auto supported_ops = GetNnapiSupportedOps();

  auto test_model = [&](const std::string& model_uri, int get_capability_ver, bool optimize, bool stop_at_nms,
                        PartitionStats& stats) {
    // make sure we can't save a model with compiled ops. input/output model format doesn't matter
    SessionOptions so;
    if (optimize) {
      so.graph_optimization_level = TransformerLevel::Level3;

    } else {
      so.graph_optimization_level = TransformerLevel::Level1;
    }

    auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

    if (optimize) {
      session->FilterEnabledOptimizers({"NchwcTransformer"});
    }

    const bool print_orders = true;
    ASSERT_STATUS_OK(session->RegisterExecutionProvider(
        std::make_unique<InternalTestingExecutionProvider>(supported_ops, get_capability_ver,
                                                           print_orders, stop_at_nms)));

    ASSERT_STATUS_OK(session->Load(model_uri));
    const auto& graph = session->GetGraph();
    GraphViewer viewer{graph};

    // save node count before optimization/partitioning
    const auto num_nodes = graph.NumberOfNodes();

    ASSERT_STATUS_OK(session->Initialize());

    const auto& func_mgr = session->GetSessionState().GetFuncMgr();
    NodeComputeInfo* compute_func = nullptr;

    stats.num_nodes_not_handled = 0;
    stats.num_compiled_nodes = 0;

    std::unordered_set<const Node*> post_nms_nodes;
    if (stop_at_nms) {
      for (const auto& node : graph.Nodes()) {
        if (node.OpType() == "NonMaxSuppression") {
          post_nms_nodes.insert(&node);

          // add all the downstream nodes to post_nms_nodes
          std::queue<const Node*> nodes_to_add;
          nodes_to_add.push(&node);
          while (!nodes_to_add.empty()) {
            const Node* cur_node = nodes_to_add.front();
            nodes_to_add.pop();

            std::for_each(cur_node->OutputNodesBegin(), cur_node->OutputNodesEnd(),
                          [&nodes_to_add, &post_nms_nodes](const Node& output_node) {
                            nodes_to_add.push(&output_node);
                            post_nms_nodes.insert(&output_node);
                          });
          }
        }
      }
    }

    for (const auto& node : graph.Nodes()) {
      if (!stop_at_nms || post_nms_nodes.find(&node) == post_nms_nodes.cend()) {
        EXPECT_EQ(supported_ops.count(node.OpType()), size_t(0))
            << "Nodes with supported op types should have been replaced. Node with type "
            << node.OpType() << " was not.";
      } else {
        EXPECT_NE(node.GetExecutionProviderType(), utils::kInternalTestingExecutionProvider)
            << "Node is downstream from an NonMaxSuppression node and should not have been taken. Node:"
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
  };

  std::vector<std::string> model_paths = {
      //"D://tflite_models//mlperf_models//deeplabv3_mnv2_ade20k_float.onnx",
      //"D://tflite_models//mlperf_models//deeplabv3_mnv2_ade20k_float-int8.onnx",
      //"D://tflite_models//mlperf_models//mobilebert.onnx",
      //"D://tflite_models//mlperf_models//mobilebert-int8.onnx",
      //"D://tflite_models//mlperf_models//mobiledet.onnx",
      //"D://tflite_models//mlperf_models//mobiledet-int8.onnx",
      //"D://tflite_models//mlperf_models//mobilenet_edgetpu_224_1.0_float.onnx",
      //"D://tflite_models//mlperf_models//mobilenet_edgetpu_224_1.0_float-int8.onnx",
      //"D://tflite_models//mlperf_models//mobilenet_v1.1.0_224_opset12.onnx",
      //"D://tflite_models//mlperf_models//pytorch.mobilenet_v2_float.onnx",
      //"D://tflite_models//mlperf_models//pytorch.mobilenet_v2_uint8.onnx",
      //"D://tflite_models//mlperf_models//resnet50_v1.onnx",
      //"D://tflite_models//mlperf_models//resnet50_v1-int8.onnx",
      "D://tflite_models//mlperf_models//ssd_mobilenet_v2_300_float.onnx",
      //      "D://tflite_models//mlperf_models//ssd_mobilenet_v2_300_float-int8.onnx"
      // "C:\\Users\\scmckay\\Downloads\\yolov4\\yolov4.onnx",
  };

  // for list of mlperf models
  for (const auto model_uri : model_paths) {
    PartitionStats old_stats{}, new_stats{}, stop_at_nms_stats{};

    std::cout << "Model:" << model_uri << std::endl;

    bool optimize = false;
    bool stop_at_nms = false;
    // test_model(model_uri, 1, optimize, stop_at_nms, old_stats);

    //std::cout << "\n\nVersion 2\n";
    //test_model(model_uri, 2, optimize, stop_at_nms, new_stats);

    std::cout << "\n\nVersion 3\n";
    test_model(model_uri, 3, optimize, stop_at_nms, new_stats);

    std::cout << "\n\nStop at NMS\n";
    test_model(model_uri, 3, optimize, stop_at_nms = true, new_stats);
    /*

    stop_at_nms = true;
    test_model(model_uri, true, optimize, stop_at_nms, stop_at_nms_stats);

    std::cout << " Partitions:old=" << old_stats.num_compiled_nodes << ", new=" << new_stats.num_compiled_nodes
              << ", stop_at_nms=" << stop_at_nms_stats.num_compiled_nodes << std::endl;
    std::cout << " NodesHandled:old=" << old_stats.num_nodes_handled << ", new=" << new_stats.num_nodes_handled
              << ", stop_at_nms=" << stop_at_nms_stats.num_compiled_nodes << std::endl;

    // shouldn't change the nodes that are handled
    ASSERT_EQ(old_stats.num_nodes_not_handled, new_stats.num_nodes_not_handled);
    */

    //optimize = true;
    //std::cout << " optimized:\n";
    //test_model(model_uri, false, optimize, old_stats);
    //test_model(model_uri, true, optimize, new_stats);

    //std::cout << " Partitions:old=" << old_stats.num_compiled_nodes << ", new=" << new_stats.num_compiled_nodes
    //          << std::endl;
    //std::cout << " NodesHandled:old=" << old_stats.num_nodes_handled << ", new=" << new_stats.num_nodes_handled
    //          << std::endl;

    //// shouldn't change the nodes that are handled
    //ASSERT_EQ(old_stats.num_nodes_not_handled, new_stats.num_nodes_not_handled);
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
