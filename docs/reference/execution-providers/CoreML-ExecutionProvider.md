---
title: CoreML
parent: Execution Providers
grand_parent: Reference
---
{::options toc_levels="2" /}

# CoreML Execution Provider

[CoreML](https://developer.apple.com/documentation/coreml) is a unified interface to CPU, GPU, and Neural Engine accelerators on macOS and iOS.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements

The CoreML Execution Provider (EP) requires macOS 10.13 or higher, or iOS 11.0 or higher.

## Build

**FIXME** add desktop info

If using CoreML on iOS, please see the [ONNX Runtime Mobile](../../how-to/mobile) deployment information.

## Usage

The ONNX Runtime API details are [here](../api). The CoreML EP can we used via the C, C++ or Objective-C APIs

The CoreML EP must be explicitly registered when creating the inference session. For example:

```C++
Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
Ort::SessionOptions so;
uint32_t coreml_flags = 0;
Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(so, coreml_flags));
Ort::Session session(env, model_path, sf);
```

## Configuration Options

There are several run time options available for the CoreML EP.

To use the CoreML EP run time options, create an unsigned integer representing the options, and set each individual options by using the bitwise OR operator.

```
uint32_t coreml_flags = 0;
coreml_flags |= COREML_FLAG_USE_CPU_ONLY;
```

### Available Options

##### COREML_FLAG_USE_CPU_ONLY

Limit CoreML to running on CPU only.

This may decrease the performance but will provide reference output value without precision loss, which is useful for validation.

#####  COREML_FLAG_ENABLE_ON_SUBGRAPH

Enable CoreML EP to run on subgraphs. **FIXME** Define what a subgraph is. Is it Loop/Scan/If?


##### COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE

By default the CoreML EP will be enabled for all compatible Apple devices.

Setting this option will only enable CoreML EP for Apple devices with the Apple Neural Engine (ANE). 
Enabling this option will not however guarantee the entire model to be executed using ANE, as it will depend on whether ANE supports all the operators used in the model.

