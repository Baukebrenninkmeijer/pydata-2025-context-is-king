"""
Configuration management using Pydantic Settings
"""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .types import PipelineConfig


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file
    """

    # Processing
    chunk_size: int = Field(default=200, description="Size of text chunks in characters")
    chunk_overlap: int = Field(default=40, description="Overlap between chunks in characters")

    # Embedding
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="SentenceTransformer embedding model")
    embedding_device: str = Field(default="cuda", description="Device for embedding model (cuda/cpu)")
    embedding_batch_size: int = Field(default=32, description="Batch size for embedding generation")

    # Storage
    chroma_db_path: str = Field(default="./chroma_db", description="Path to ChromaDB persistent storage")

    # Retrieval
    default_k: int = Field(default=10, description="Default number of chunks to retrieve")
    similarity_threshold: float = Field(default=0.0, description="Minimum similarity threshold for retrieval")

    # Reranking
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder model for reranking"
    )
    reranker_device: str = Field(default="auto", description="Device for reranker model (auto/cuda/mps/cpu)")
    reranker_batch_size: int = Field(default=16, description="Batch size for reranking")

    # Context
    max_context_length: int = Field(default=32000, description="Maximum context length in tokens")

    # APIs
    nvidia_model: str = Field(default="moonshotai/kimi-k2-instruct", description="NVIDIA API model for generation")
    nvidia_rate_limit: int = Field(default=40, description="NVIDIA API rate limit (requests per minute)")
    judge_model: str = Field(default="gpt-4.1-2025-04-14", description="OpenAI model for judge evaluation")

    # API Keys
    nvidia_api_key: Optional[str] = Field(default=None, description="NVIDIA API key")
    openai_api_key: Optional[str] = Field(default=None, alias="ORQ_API_KEY", description="OpenAI/ORQ API key")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to PipelineConfig object"""
        return PipelineConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_model=self.embedding_model,
            embedding_device=self.embedding_device,
            embedding_batch_size=self.embedding_batch_size,
            chroma_db_path=self.chroma_db_path,
            default_k=self.default_k,
            similarity_threshold=self.similarity_threshold,
            reranker_model=self.reranker_model,
            reranker_device=self.reranker_device,
            reranker_batch_size=self.reranker_batch_size,
            max_context_length=self.max_context_length,
            nvidia_model=self.nvidia_model,
            nvidia_rate_limit=self.nvidia_rate_limit,
            judge_model=self.judge_model,
            nvidia_api_key=self.nvidia_api_key,
            openai_api_key=self.openai_api_key,
        )


def load_config() -> PipelineConfig:
    """Load configuration from environment variables and .env file"""
    settings = Settings()
    return settings.to_pipeline_config()


def load_config_from_dict(config_dict: dict) -> PipelineConfig:
    """Load configuration from dictionary"""
    settings = Settings(**config_dict)
    return settings.to_pipeline_config()


def validate_config(config: PipelineConfig) -> dict:
    """Validate pipeline configuration and return validation results"""
    validation = {"valid": True, "errors": [], "warnings": []}

    # Check required API keys
    if not config.nvidia_api_key:
        validation["errors"].append("NVIDIA_API_KEY is required")
        validation["valid"] = False

    if not config.openai_api_key:
        validation["errors"].append("OPENAI_API_KEY is required")
        validation["valid"] = False

    # Check device availability
    if config.embedding_device in ["cuda", "auto"] or config.reranker_device in ["cuda", "auto"]:
        try:
            import torch

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            if not cuda_available and not mps_available:
                validation["warnings"].append("No GPU acceleration available (CUDA/MPS), falling back to CPU")
            elif mps_available and not cuda_available:
                validation["warnings"].append("Using MPS (Apple Silicon) acceleration")
            elif cuda_available:
                validation["warnings"].append("Using CUDA acceleration")
        except ImportError:
            validation["warnings"].append("PyTorch not installed, cannot check GPU availability")

    # Check paths
    if not os.path.exists(os.path.dirname(config.chroma_db_path)):
        try:
            os.makedirs(os.path.dirname(config.chroma_db_path), exist_ok=True)
        except Exception as e:
            validation["errors"].append(f"Cannot create ChromaDB directory: {e}")
            validation["valid"] = False

    # Check numerical parameters
    if config.chunk_size <= 0:
        validation["errors"].append("chunk_size must be positive")
        validation["valid"] = False

    if config.chunk_overlap >= config.chunk_size:
        validation["errors"].append("chunk_overlap must be less than chunk_size")
        validation["valid"] = False

    if config.max_context_length <= 0:
        validation["errors"].append("max_context_length must be positive")
        validation["valid"] = False

    if config.nvidia_rate_limit <= 0:
        validation["errors"].append("nvidia_rate_limit must be positive")
        validation["valid"] = False

    return validation
