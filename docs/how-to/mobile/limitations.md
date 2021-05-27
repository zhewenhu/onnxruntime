---
title: Limitations
parent: Deploy ONNX Runtime Mobile
grand_parent: How to
nav_order: 7
---

## Limitations

A minimal build has the following limitations currently:
  - No support for ONNX format models
    - Model must be converted to ORT format
  - No support for runtime optimizations
    - Optimizations should be performed prior to conversion to ORT format
  - Execution providers that statically register kernels (e.g. ONNX Runtime CPU Execution Provider) are supported by default
  - Limited support for runtime partitioning (assigning nodes in a model to specific execution providers)
    - Execution providers that statically register kernels and will be used at runtime MUST be enabled when creating the ORT format model
    - Execution providers that compile nodes are optionally supported, and nodes they create will be correctly partitioned at runtime
      - currently this is limited to the NNAPI Execution Provider

We do not currently offer backwards compatibility guarantees for ORT format models, as we will be expanding the capabilities in the short term and may need to update the internal format in an incompatible manner to accommodate these changes. You may need to regenerate the ORT format models to use with a future version of ONNX Runtime. Once the feature set stabilizes we will provide backwards compatibility guarantees.