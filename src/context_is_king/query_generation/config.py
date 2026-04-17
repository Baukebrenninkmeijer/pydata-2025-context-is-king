"""Configuration for WikiText Query Generation Pipeline"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WikiTextConfig:
    """Configuration for WikiText processing pipeline."""

    # Data paths
    data_dir: Path = Path("data")
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    chroma_db_path: Path = field(default_factory=lambda: Path("data/vector_stores/chroma_db"))

    # Dataset configuration
    local_data_file: Path = field(default_factory=lambda: Path("data/processed/nq_question_answer.parquet"))
    sample_size: int | None = None  # None for full dataset (limits documents BEFORE chunking)
    sample_chunks: int | None = None  # None for all chunks (limits chunks AFTER text splitting)

    # Text processing
    chunk_size: int = 1900
    chunk_overlap: int = 200
    max_token_length: int = 2000

    # Embedding configuration
    embedding_model: str = "azure/text-embedding-3-small"
    embedding_dimensions: int = 768  # Reduced from default 1536 for efficiency
    embedding_batch_size: int = 100
    embedding_checkpoint_interval: int = 50_000  # Save progress every N chunks

    # LLM filtering configuration
    filter_model: str = "azure/gpt-4.1-mini"
    
    # Consecutive chunk processing
    consecutive_chunks_count: int = 3  # Number of consecutive chunks to process together
    
    # Question generation and evaluation
    question_generation_model: str = "azure/gpt-4.1-mini"
    question_evaluation_model: str = "azure/gpt-4.1-mini"

    # Rate limiting (increased for better throughput)
    max_concurrent_requests: int = 100
    requests_per_minute: int = 1000
    tokens_per_minute: int = 600_000
    request_timeout: float = 120.0

    # Batch processing
    llm_batch_max_requests: int = 1000
    filter_batch_size: int = 20  # Documents per filtering batch (reduced to prevent timeouts)
    filter_samples_per_doc: int | None = None  # Max chunks per document for filtering
    filter_max_total_chunks: int | None = None  # Max total chunks for filtering
    sample_chunks: int | None = None  # Limit total chunks after splitting

    # API configuration
    orq_api_key: str = field(default_factory=lambda: os.getenv("ORQ_API_KEY", ""))
    orq_base_url: str = "https://api.orq.ai/v2/proxy"

    # Logging
    log_level: str = "INFO"

    @property
    def dataset_name(self) -> str:
        """Generate dataset name from local data file."""
        return self.local_data_file.stem  # Gets filename without extension

    def __post_init__(self):
        """Validate configuration and create directories."""
        if not self.orq_api_key:
            raise ValueError("ORQ_API_KEY environment variable is required")

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate batch sizes
        if self.embedding_batch_size > self.embedding_checkpoint_interval:
            raise ValueError("embedding_batch_size cannot exceed embedding_checkpoint_interval")

        # Validate consecutive chunks configuration
        if self.consecutive_chunks_count < 1:
            raise ValueError("consecutive_chunks_count must be at least 1")


@dataclass
class ProcessingStage:
    """Track processing stage state."""

    name: str
    completed: bool = False
    output_file: Path | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineState:
    """Track overall pipeline state and progress."""

    config: WikiTextConfig
    stages: list[ProcessingStage] = field(default_factory=list)
    total_documents: int = 0
    processed_documents: int = 0
    filtered_documents: int = 0
    embedded_documents: int = 0

    def __post_init__(self):
        """Initialize processing stages."""
        if not self.stages:
            self.stages = [
                ProcessingStage("data_loading"),
                ProcessingStage("data_validation"),
                ProcessingStage("chunking"),
                ProcessingStage("embedding_generation"),
                ProcessingStage("document_filtering"),
                ProcessingStage("chroma_ingestion"),
            ]

    def get_stage(self, name: str) -> ProcessingStage | None:
        """Get processing stage by name."""
        return next((stage for stage in self.stages if stage.name == name), None)

    def mark_stage_complete(self, name: str, output_file: Path | None = None, **metadata):
        """Mark a processing stage as complete."""
        stage = self.get_stage(name)
        if stage:
            stage.completed = True
            stage.output_file = output_file
            stage.metadata.update(metadata)

    def is_stage_complete(self, name: str) -> bool:
        """Check if a processing stage is complete."""
        stage = self.get_stage(name)
        return stage.completed if stage else False

    def get_progress_summary(self) -> dict:
        """Get summary of processing progress."""
        completed_stages = sum(1 for stage in self.stages if stage.completed)
        return {
            "stages_completed": completed_stages,
            "total_stages": len(self.stages),
            "progress_percent": (completed_stages / len(self.stages)) * 100,
            "total_documents": self.total_documents,
            "processed_documents": self.processed_documents,
            "filtered_documents": self.filtered_documents,
            "embedded_documents": self.embedded_documents,
        }
