// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <unordered_set>

#include "core/graph/basic_types.h"

namespace onnxruntime {
struct ComputeCapability;
class GraphViewer;
class NodeArg;
class Node;

namespace utils {

///**
//Perform a topological sort that groups nodes that are supported/unsupported together.
//This enables better grouping to minimize the partitions handled by the execution provider.
//
//@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
//@param supported_nodes Set of nodes that the execution provider wants to handle.
//@returns Ordered indexes of the sorted nodes.
//
//TODO: Should the input supported nodes be const Node* or NodeIndex values?
//*/
//std::vector<NodeIndex> PartitionAwareTopoSort(const GraphViewer& graph_viewer,
//                                              const std::unordered_set<const Node*>& supported_nodes);
//
///**
//Create set of nodes that are excluded due to operations we stop considering nodes for the execution provider at.
//e.g. a NonMaxSuppression node is the beginning of post-processing in SSD models, and it's cheaper to exclude the
//NonMaxSuppression and all downstream nodes from the NNAPI EP and run them on CPU.
//All nodes with the operators in stop_ops, and their downstream nodes, will be added to the set returned.
//
//@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
//@param stop_ops Set of operator names at which we stop considering nodes for assignment to this execution provider.
//@returns Set of nodes to exclude
//
//TODO: Should this return const Node* or NodeIndex values?
//*/
//std::unordered_set<const Node*> CreateExcludedNodeSet(const GraphViewer& graph_viewer,
//                                                      const std::unordered_set<std::string>& stop_ops);
//

/**
Create a ComputeCapability instance from the group of nodes.
Will determine the inputs and outputs required

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param group Group of nodes to include in the ComputeCapability
@param generate_metadef_name Functor to create the name for the MetaDef.
                             Most likely should call IExecutionProvider::GenerateMetaDefId.
                             See onnxruntime/test/providers/internal_testing/internal_testing_execution_provider.cc for
                             example usage.
@param execution_provider_name Name of execution provider creating the ComputeCapability instance.

@returns New ComputeCapability instance.
*/
std::unique_ptr<ComputeCapability> MakeComputeCapability(const GraphViewer& graph_viewer,
                                                         const std::vector<const Node*>& group,
                                                         const std::function<std::string()>& generate_metadef_name,
                                                         const std::string& execution_provider_name);

/** 
Create the supported partitions for the execution provider. 

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param supported_nodes Set of nodes that the execution provider wants to handle. 
@param stop_ops Set of operator names at which we stop considering nodes for assignment to this execution provider.
@param generate_metadef_name Functor to create the name for the MetaDef. 
                             Most likely should call IExecutionProvider::GenerateMetaDefId. 
                             See onnxruntime/test/providers/internal_testing/internal_testing_execution_provider.cc
                             for example usage.
@param execution_provider_name Name of execution provider creating the ComputeCapability instance.
@param debug_output Print diagnostic output about the partitions and reasons for partition breaks. 
                    No-op in a release build.

@returns ComputeCapability instances for all partitions assigned to the execution provider. 
*/
std::vector<std::unique_ptr<ComputeCapability>> CreateSupportedPartitions(
    const GraphViewer& graph_viewer,
    const std::unordered_set<const Node*>& supported_nodes,
    const std::unordered_set<std::string>& stop_ops,
    const std::function<std::string()>& generate_metadef_name,
    const std::string& execution_provider_name,
    bool debug_output = false);

}  // namespace utils
}  // namespace onnxruntime
