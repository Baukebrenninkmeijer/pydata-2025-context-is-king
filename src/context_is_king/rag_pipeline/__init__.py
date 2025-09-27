"""
RAG Pipeline for PyData 2025 Context Is King Research Project
"""

from pathlib import Path

from loguru import logger

from .context_assembly.engine import ContextAssembly
from .document_processing.processor import DocumentProcessor
from .embedding_storage.manager import EmbeddingStorageManager
from .generation.interface import GenerationInterface
from .orchestration.controller import ExperimentController
from .query_enhancement import QueryRewriter, DualRetrieval
from .reranking.module import RerankerModule
from .retrieval.engine import RetrievalEngine

__all__ = [
    "ContextAssembly",
    "DocumentProcessor",
    "EmbeddingStorageManager",
    "ExperimentController",
    "GenerationInterface",
    "QueryRewriter",
    "DualRetrieval",
    "RerankerModule",
    "RetrievalEngine",
]


DATA_DIR = Path(__file__).parents[2] / "data"
logger.debug(f"{DATA_DIR=}")
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
