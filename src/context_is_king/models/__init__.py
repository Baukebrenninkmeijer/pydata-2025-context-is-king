"""
Model Interface Module

Provides unified interfaces for interacting with various language models
through different APIs and providers.
"""

from .interface import ModelInterface, ModelConfig, QueryResult

__all__ = ["ModelInterface", "ModelConfig", "QueryResult"]