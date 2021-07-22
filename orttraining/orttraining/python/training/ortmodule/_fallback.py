# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _logger

import torch
import warnings

from enum import IntFlag
from typing import Optional


class _FallbackPolicy(IntFlag):
    '''Policy to trigger fallback from ONNX Runtime engine to PyTorch

    Each policy can be combined with the others (using |) in order to aggregate them'''

    FALLBACK_DISABLE = 1
    FALLBACK_FORCE_TORCH_FORWARD = 2
    FALLBACK_FORCE_TORCH_BACKWARD = 4
    FALLBACK_UNSUPPORTED_DEVICE = 8
    FALLBACK_UNSUPPORTED_INPUT = 16
    FALLBACK_UNSUPPORTED_OUTPUT = 32
    FALLBACK_UNSUPPORTED_TORCH_MODEL = 64
    FALLBACK_UNSUPPORTED_ONNX_MODEL = 128
    FALLBACK_BAD_INITIALIZATION = 256

    def is_set(self, policy):
        '''Check whether `policy` is set on the `_FallbackPolicy instance

        FALLBACK_DISABLE implies the check will always fail and return False
        '''

        return not _FallbackPolicy.is_disabled(self) and policy in self

    def is_disabled(self):
        '''Check whether `_FallbackPolicy.FALLBACK_DEVICE is set on the `_FallbackPolicy instance'''

        return _FallbackPolicy.FALLBACK_DISABLE in self


class ORTModuleFallbackException(Exception):
    '''Base exception class for fallback

    Although it must be specialized for specific scenarios,
    it can also be used for generic exception that require fallback
    '''
    pass


class ORTModuleInitException(ORTModuleFallbackException):
    '''Trigger fallback for ORTModule initialization related exceptions

    This exception is triggered when an incompatible or missing requirements for ORTModule are detected,
    including PyTorch version, missing ORTModule's PyTorch C++ extension binaries, etc.
    '''
    pass

class ORTModuleDeviceException(ORTModuleFallbackException):
    '''Trigger fallback for device related exceptions

    NOTE: This exception is raised during device validation within ORTModule frontend.
    Some device related exceptions can only be detected during PyTorch ONNX exporter execution.
    This exception does not capture these scenarios.
    '''

    pass


class ORTModuleIOError(ORTModuleFallbackException):
    '''Trigger fallback for I/O related exceptions

    NOTE: This exception is raised during I/O validation within ORTModule Frontend.
    Some I/O related exceptions can only be detected during PyTorch ONNX exporter execution.
    This exception does not capture these scenarios.
    '''

    pass


class ORTModuleTorchModelException(ORTModuleFallbackException):
    '''Trigger fallback for PyTorch modules related exceptions

    This exception is raised during model validation within ORTModule frontend and is based on
    checking type(model) over a hardcoded list of incompatible models.
    '''

    pass


class ORTModuleONNXModelException(ORTModuleFallbackException):
    '''Trigger fallback for ONNX model related exceptions

    This exception is raised during model conversion to ONNX and post-processing validation within ORTModule frontend.
    '''

    pass

class _FallbackManager(object):
    '''Manages fallbacks based on incoming exceptions and specified policies

    The basic algorithm is based on a dictionary whose keys are the supported fallback policies
    and and values are a set of Exception that must be detected.

    When an exception that matches one of the enabled policies are detected,
    a fallback will be pending to execute by ORTModule frontend.

    On the other hand, when the exception doesn't match any enabled policy, the exception will
    be raised to the user, terminating execution
    '''

    def __init__(self,
                 policy: _FallbackPolicy,
                 log_level: _logger.LogLevel):
        super().__init__()
        self._log_level = log_level
        self._policy_exception_map = {_FallbackPolicy.FALLBACK_FORCE_TORCH_FORWARD.value: {ORTModuleFallbackException,
                                                                                           ORTModuleDeviceException,
                                                                                           ORTModuleIOError,
                                                                                           ORTModuleTorchModelException,
                                                                                           ORTModuleONNXModelException},
                                      _FallbackPolicy.FALLBACK_FORCE_TORCH_BACKWARD.value: {ORTModuleFallbackException,
                                                                                            ORTModuleDeviceException,
                                                                                            ORTModuleIOError,
                                                                                            ORTModuleTorchModelException,
                                                                                            ORTModuleONNXModelException},
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE.value: {ORTModuleDeviceException},
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_INPUT.value: {ORTModuleIOError},
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_OUTPUT.value: {ORTModuleIOError},
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_TORCH_MODEL.value : {ORTModuleTorchModelException},
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_ONNX_MODEL.value : {ORTModuleONNXModelException},
                                      _FallbackPolicy.FALLBACK_BAD_INITIALIZATION.value : {ORTModuleInitException},
                                      }
        self._policy = policy
        self._exception = None

    def _handle_exception(self,
                          exception: Exception,
                          policy: Optional[_FallbackPolicy] = None) -> None:

        def _set_exception(policy, exception):
            '''Sets `exception` into `_FallbackManager` based on the specified `policy`

            If the incoming `exception` is handled by the specified `policy`, than `_FallbackManager`
            will save the exception as context so that ORTModule can learn about a pending fallback
            and trigger it during model execution.

            Args:
                policy (_FallbackPolicy or None): Policy to be checked for the incoming `exception`.
                    if None is specified, all (except _FallbackPolicy.FALLBACK_DISABLE) are implicitly checked
                exception (ORTModuleFallbackException): Exception that must be handled

            Raises:
                Exception: when there is no matching `policy` for the incoming `exception`
            '''

            if policy is not _FallbackPolicy.FALLBACK_DISABLE and \
                    self._policy.is_set(policy) and \
                    (policy.value in self._policy_exception_map and type(exception) in self._policy_exception_map[policy.value]):

                if self._log_level <= _logger.LogLevel.WARNING:
                    warnings.warn(f'Fallback for policy {policy.name} is now pending.', UserWarning)
                self._exception = exception

        if policy is None:
            for policy in _FallbackPolicy:
                _set_exception(policy, exception)
        else:
            _set_exception(policy, exception)

        if self._exception is None:
            raise exception

    def _is_pending(self) -> bool:
        '''Returns True when a fallback is pending

        ORTModule must execute fallback to PyTorch engine when a pending fallback is detected
        '''

        return self._exception is not None

    def _fallback(self, model: torch.nn.Module, *inputs, **kwargs):
        '''Executes user PyTorch `model` using the provided inputs and return the result'''

        if self._log_level <= _logger.LogLevel.WARNING:
            warnings.warn(f'Fallback due to exception {type(self._exception)} was triggered.', UserWarning)
        return model(*inputs, **kwargs)

    @staticmethod
    def raise_exception(new_exception: ORTModuleFallbackException, raised_exception: Exception) -> ORTModuleFallbackException:
        '''Raises `new_exception` and set `raised_exception` as its cause'''

        raise new_exception(raised_exception) from raised_exception