"""Embedding generation with rate limiting and batch processing."""

from typing import Any

import polars as pl
from loguru import logger
from openai import OpenAI as OpenAIClient
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .config import WikiTextConfig

# Initialize rich console for progress tracking
console = Console()


class EmbeddingGenerator:
    """Generate embeddings for documents with batch processing and rate limiting."""

    def __init__(self, config: WikiTextConfig):
        self.config = config
        self.client = OpenAIClient(api_key=config.orq_api_key, base_url=config.orq_base_url)

        # Setup logging
        logger.configure(
            handlers=[
                {
                    "sink": "logs/embedding_generation.log",
                    "level": config.log_level,
                    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                    "rotation": "10 MB",
                }
            ]
        )

    def generate_embeddings_batch(self, texts: list[str], model: str, dimensions: int = None) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed
            model: Model name for embeddings
            dimensions: Optional number of dimensions to truncate to

        Returns:
            List of embedding vectors
        """
        try:
            # Use dimensions parameter if provided (for text-embedding-3-* models)
            kwargs = {"model": model, "input": texts}
            if dimensions is not None:
                kwargs["dimensions"] = dimensions
            
            response = self.client.embeddings.create(**kwargs)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero vectors as fallback with correct dimensions
            fallback_dims = dimensions or self.config.embedding_dimensions
            return [[0.0] * fallback_dims for _ in texts]

    def generate_embeddings_in_batches(
        self, documents_df: pl.DataFrame, text_column: str = "chunked_prompt", id_column: str = "unique_id"
    ) -> pl.DataFrame:
        """Generate embeddings for all documents in batches.

        Args:
            documents_df: Polars DataFrame with documents
            text_column: Column name containing text to embed
            id_column: Column name containing unique identifiers

        Returns:
            DataFrame with embeddings added
        """
        logger.info(f"Starting embedding generation for {len(documents_df)} documents")

        # Convert to lists for batch processing
        texts = documents_df.select(text_column).to_series().to_list()
        ids = documents_df.select(id_column).to_series().to_list()

        all_embeddings = []
        batch_size = self.config.embedding_batch_size
        checkpoint_interval = self.config.embedding_checkpoint_interval

        # Process in large batches to manage memory with cancellation checks
        total_large_batches = (len(texts) + checkpoint_interval - 1) // checkpoint_interval

        # Create a dedicated console for embedding progress to avoid conflicts
        embedding_console = Console(stderr=True, force_terminal=True)

        # Use Rich Progress with dedicated console to avoid conflicts with other progress bars
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=embedding_console,
            transient=True,  # Progress bar disappears when complete
        ) as progress:
            embedding_task = progress.add_task(
                f"Generating embeddings ({total_large_batches} large batches)", total=len(texts)
            )

            for large_batch_idx, large_batch_start in enumerate(range(0, len(texts), checkpoint_interval)):
                # Check for KeyboardInterrupt
                try:
                    large_batch_end = min(large_batch_start + checkpoint_interval, len(texts))
                    large_batch_texts = texts[large_batch_start:large_batch_end]
                    large_batch_ids = ids[large_batch_start:large_batch_end]

                    # Update progress description for current batch
                    progress.update(
                        embedding_task,
                        description=f"Generating embeddings (batch {large_batch_idx + 1}/{total_large_batches})",
                    )

                    # Process large batch in smaller API batches
                    batch_embeddings = []
                    total_api_batches = (len(large_batch_texts) + batch_size - 1) // batch_size

                    for api_batch_idx, batch_start in enumerate(range(0, len(large_batch_texts), batch_size)):
                        # Check for KeyboardInterrupt in inner loop
                        try:
                            batch_end = min(batch_start + batch_size, len(large_batch_texts))
                            batch_texts = large_batch_texts[batch_start:batch_end]

                            embeddings = self.generate_embeddings_batch(
                                batch_texts, 
                                self.config.embedding_model, 
                                dimensions=self.config.embedding_dimensions
                            )
                            batch_embeddings.extend(embeddings)

                            # Update progress for each API batch completed
                            progress.update(embedding_task, advance=len(batch_texts))
                        except KeyboardInterrupt:
                            logger.info("Embedding generation interrupted by user")
                            raise

                except KeyboardInterrupt:
                    logger.info("Embedding generation interrupted by user - saving partial results")
                    raise

                # Save intermediate results
                large_batch_df = documents_df.filter(pl.col(id_column).is_in(large_batch_ids)).with_columns(
                    pl.Series("embedding", batch_embeddings)
                )

                # Append to Delta table for incremental processing with atomic commits
                output_file = self.config.processed_dir / "embeddings_incremental.delta"
                if large_batch_start == 0:
                    large_batch_df.write_delta(str(output_file), mode="overwrite")
                else:
                    # Delta handles append natively with atomic commits
                    large_batch_df.write_delta(str(output_file), mode="append")

                all_embeddings.extend(batch_embeddings)

        # Return final DataFrame with embeddings
        result_df = documents_df.with_columns(pl.Series("embedding", all_embeddings))

        logger.info(f"Generated embeddings for {len(result_df)} documents")
        return result_df

    def load_or_generate_embeddings(self, documents_df: pl.DataFrame, force_regenerate: bool = False) -> pl.DataFrame:
        """Load existing embeddings or generate new ones.

        Args:
            documents_df: DataFrame with documents to embed
            force_regenerate: If True, regenerate embeddings even if they exist

        Returns:
            DataFrame with embeddings
        """
        embeddings_file = self.config.processed_dir / "document_embeddings.delta"

        if not force_regenerate and embeddings_file.exists():
            logger.info(f"Found existing embeddings at {embeddings_file}")
            logger.info("Loading existing embeddings (checking file size for optimal loading strategy)...")
            try:
                # Smart loading: use streaming for large files, direct read for small ones
                total_size_bytes = sum(f.stat().st_size for f in embeddings_file.rglob("*") if f.is_file())
                size_mb = total_size_bytes / (1024 * 1024)

                if size_mb > 100:  # > 100MB use streaming
                    logger.info(
                        f"Large embedding file ({size_mb:.1f}MB) - using streaming engine for memory efficiency"
                    )
                    return pl.scan_delta(str(embeddings_file)).collect(engine="streaming")
                logger.info(f"Small embedding file ({size_mb:.1f}MB) - using direct read")
                return pl.read_delta(str(embeddings_file))
            except Exception as e:
                logger.warning(f"Could not load embeddings: {e}. Regenerating...")

        # Generate new embeddings
        logger.info("No existing embeddings found - Starting fresh embedding generation")
        logger.info(f"Embedding model: {self.config.embedding_model}")
        result_df = self.generate_embeddings_in_batches(documents_df)

        # Save final embeddings as Delta with atomic commit
        result_df.write_delta(str(embeddings_file), mode="overwrite")
        logger.info(f"Saved embeddings to {embeddings_file}")

        return result_df

    def validate_embeddings(self, embeddings_df: pl.DataFrame) -> dict[str, Any]:
        """Validate embedding quality and consistency.

        Args:
            embeddings_df: DataFrame with embeddings

        Returns:
            Validation metrics
        """
        logger.info("Validating embeddings")

        # Check for zero vectors (indicating failures)
        zero_vectors = 0
        embedding_dims = None

        for embedding in embeddings_df.select("embedding").to_series():
            if embedding_dims is None:
                embedding_dims = len(embedding)

            if all(x == 0.0 for x in embedding):
                zero_vectors += 1

        total_embeddings = len(embeddings_df)

        validation_results = {
            "total_embeddings": total_embeddings,
            "zero_vectors": zero_vectors,
            "zero_vector_percentage": (zero_vectors / total_embeddings) * 100,
            "embedding_dimensions": embedding_dims,
            "valid_embeddings": total_embeddings - zero_vectors,
        }

        logger.info(f"Embedding validation: {validation_results}")

        if zero_vectors > total_embeddings * 0.05:  # More than 5% failures
            logger.warning(
                f"High failure rate: {zero_vectors} zero vectors ({validation_results['zero_vector_percentage']:.1f}%)"
            )

        return validation_results
