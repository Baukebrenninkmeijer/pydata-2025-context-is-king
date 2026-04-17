"""
Core data types for the RAG pipeline using Pydantic models
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator
import os


class Document(BaseModel):
    """Represents a source document"""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_id: str = ""
    source: str = ""
    doc_type: str = ""  # 'paul_graham', 'arxiv', 'conversation'

    @field_validator("doc_id", mode="before")
    @classmethod
    def set_doc_id(cls, v: str, info) -> str:
        if not v and "source" in info.data and info.data["source"]:
            return Path(info.data["source"]).stem
        return v


class DocumentChunk(BaseModel):
    """Represents a chunk of a document"""

    content: str
    doc_id: str | None
    chunk_id: str
    start_char: int | None
    end_char: int | None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def full_id(self) -> str:
        return f"{self.doc_id}_{self.chunk_id}"


class RetrievalResult(BaseModel):
    """Result from semantic search"""

    chunk: DocumentChunk
    similarity_score: float
    rank: int
    reranked: bool = False
    rerank_score: float | None = None

    @computed_field
    @property
    def final_score(self) -> float:
        return self.rerank_score if self.reranked and self.rerank_score else self.similarity_score


class GenerationResult(BaseModel):
    """Result from LLM generation"""

    response: str
    context: str
    query: str
    model: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] | None = None
    latency_ms: float | None = None


class JudgeResult(BaseModel):
    """Result from judge evaluation"""

    is_correct: bool
    judge_response: str
    question: str
    correct_answer: str
    model_response: str
    judge_model: str
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Configuration for a single experiment"""

    experiment_id: str
    experiment_type: str  # 'needle_similarity', 'distractor_impact', etc.

    # Data sources
    needles: list[str] = Field(default_factory=list)
    haystacks: list[str] = Field(default_factory=list)
    distractors: list[str] = Field(default_factory=list)

    # Experiment variations
    retrieval_k: list[int] = Field(default_factory=lambda: [10])
    reranking: list[bool] = Field(default_factory=lambda: [True, False])
    context_types: list[str] = Field(default_factory=lambda: ["rag"])
    context_window_sizes: list[int] = Field(default_factory=lambda: [32000])

    # Processing options
    shuffle_types: list[str] = Field(default_factory=list)
    distractor_counts: list[int] = Field(default_factory=lambda: [0])

    # Evaluation
    judge_prompt_type: str = "niah"  # 'niah' or 'longmemeval'
    batch_size: int = 10

    # Output
    results_dir: str = "results/"
    save_intermediate: bool = True


class ExperimentResults(BaseModel):
    """Results from a complete experiment"""

    experiment_id: str
    config: ExperimentConfig
    generation_results: list[GenerationResult] = Field(default_factory=list)
    judge_results: list[JudgeResult] = Field(default_factory=list)
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    summary_stats: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Configuration for the entire RAG pipeline"""

    # Processing
    chunk_size: int = 1900
    chunk_overlap: int = 200

    # Embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_device: str = "mps"
    embedding_batch_size: int = 32

    # Storage
    chroma_db_path: str = "data/vector_stores/chroma_db"

    # Retrieval
    default_k: int = 10
    similarity_threshold: float = 0.0

    # Reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "auto"
    reranker_batch_size: int = 16

    # Context
    max_context_length: int = 32000

    # UI/Display
    quiet_mode: bool = False

    # APIs
    judge_model: str = "gpt-4.1"
    small_llm: str = "gpt-4.1-mini"
    large_llm: str = "gpt-4.1"

    # API Keys (loaded from environment)
    nvidia_api_key: str | None = None
    openai_api_key: str | None = os.getenv("ORQ_API_KEY")
    orq_api_key: str | None = os.getenv("ORQ_API_KEY")
    orq_base_url: str = os.environ["ORQ_BASE_URL"]

    model_config = {"extra": "forbid", "validate_assignment": True}
