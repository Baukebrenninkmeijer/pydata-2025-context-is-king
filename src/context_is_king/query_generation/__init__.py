"""WikiText Query Generation Pipeline

This module provides tools to process WikiText datasets and generate synthetic queries
for research experiments. It includes:

- Data loading and preprocessing from Hugging Face datasets
- Document chunking and tokenization
- Embedding generation with rate limiting
- LLM-based document filtering with async processing
- ChromaDB integration for vector storage

Key components:
- WikiTextProcessor: Main pipeline controller
- EmbeddingGenerator: Handles embedding creation with batching
- DocumentFilter: LLM-based document filtering with semaphore limits
- ChromaManager: Vector database operations
"""

from .processor import WikiTextProcessor
from .config import WikiTextConfig
from .embeddings import EmbeddingGenerator
from .filters import DocumentFilter
from .chroma import ChromaManager

__all__ = ["WikiTextProcessor", "WikiTextConfig", "EmbeddingGenerator", "DocumentFilter", "ChromaManager"]
