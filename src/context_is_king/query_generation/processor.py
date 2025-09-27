"""Main WikiText processing pipeline."""

import asyncio
import psutil
import re
import time

# Removed HuggingFace datasets - using local data instead
from typing import Any
from uuid import uuid4

import polars as pl
import tiktoken
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from .chroma import ChromaManager
from .config import PipelineState, WikiTextConfig
from .embeddings import EmbeddingGenerator
from .filters import DocumentFilter


class WikiTextProcessor:
    """Main pipeline for processing WikiText datasets and generating queries."""

    def __init__(self, config: WikiTextConfig):
        self.config = config
        self.state = PipelineState(config)
        self.console = Console()
        
        # Stage timing tracking
        self.stage_start_times = {}
        self.pipeline_start_time = None
        
        # Resource monitoring
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Pipeline status tracking
        self.pipeline_status = {
            1: {"name": "📥 Data Loading", "status": "⏸️ Waiting", "input_count": None, "output_count": None, "elapsed": None, "details": ""},
            2: {"name": "✅ Data Validation", "status": "⏸️ Waiting", "input_count": None, "output_count": None, "elapsed": None, "details": ""},
            3: {"name": "✂️ Document Chunking", "status": "⏸️ Waiting", "input_count": None, "output_count": None, "elapsed": None, "details": ""},
            4: {"name": "🔮 Embedding Generation", "status": "⏸️ Waiting", "input_count": None, "output_count": None, "elapsed": None, "details": ""},
            5: {"name": "🔍 LLM Filtering", "status": "⏸️ Waiting", "input_count": None, "output_count": None, "elapsed": None, "details": ""},
            6: {"name": "🗃️ ChromaDB Ingestion", "status": "⏸️ Waiting", "input_count": None, "output_count": None, "elapsed": None, "details": ""}
        }

        # Initialize components
        self.embedding_generator = EmbeddingGenerator(config)
        self.document_filter = DocumentFilter(config)
        self.chroma_manager = ChromaManager(config)

        # Initialize text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o")
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Setup logging
        logger.configure(
            handlers=[
                {
                    "sink": "logs/wikitext_processor.log",
                    "level": config.log_level,
                    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                    "rotation": "10 MB",
                }
            ]
        )

        logger.info("WikiText processor initialized")

    def _get_memory_usage(self) -> tuple[float, float]:
        """Get current memory usage in MB."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = current_memory - self.initial_memory
        return current_memory, memory_delta

    def _create_status_table(self) -> Table:
        """Create a live status table showing all pipeline stages."""
        table = Table(title="🚀 Pipeline Status Board", show_header=True, header_style="bold magenta")
        table.add_column("Stage", style="cyan", no_wrap=True, width=25)
        table.add_column("Status", style="green", width=15) 
        table.add_column("Progress", style="blue", width=20)
        table.add_column("Time", style="yellow", width=10)
        table.add_column("Details", style="dim", width=30)

        for stage_num, stage_info in self.pipeline_status.items():
            # Format progress info
            if stage_info["input_count"] and stage_info["output_count"]:
                progress_str = f"{stage_info['input_count']:,} → {stage_info['output_count']:,}"
            elif stage_info["output_count"]:
                progress_str = f"{stage_info['output_count']:,} items"
            else:
                progress_str = "—"

            # Format timing 
            time_str = f"{stage_info['elapsed']:.1f}s" if stage_info['elapsed'] else "—"

            table.add_row(
                stage_info["name"], 
                stage_info["status"], 
                progress_str,
                time_str,
                stage_info["details"][:30] + "..." if len(stage_info["details"]) > 30 else stage_info["details"]
            )

        return table

    def _start_stage(self, stage_number: int, stage_name: str, description: str = None) -> None:
        """Start a new stage with standardized logging and timing."""
        stage_key = f"stage_{stage_number}"
        self.stage_start_times[stage_key] = time.time()
        
        # Update status tracking
        self.pipeline_status[stage_number]["status"] = "⏳ Active"
        self.pipeline_status[stage_number]["details"] = description or ""
        
        logger.info("\n" + "=" * 60)
        logger.info(f"🚀 STAGE {stage_number}/6: {stage_name.upper()}")
        logger.info("=" * 60)
        if description:
            logger.info(description)

    def _complete_stage(self, stage_number: int, stage_name: str, input_count: int = None, 
                       output_count: int = None, additional_info: str = None) -> float:
        """Complete a stage with standardized logging and timing."""
        stage_key = f"stage_{stage_number}"
        elapsed = time.time() - self.stage_start_times.get(stage_key, time.time())
        
        # Update status tracking
        self.pipeline_status[stage_number]["status"] = "✅ Done"
        self.pipeline_status[stage_number]["input_count"] = input_count
        self.pipeline_status[stage_number]["output_count"] = output_count
        self.pipeline_status[stage_number]["elapsed"] = elapsed
        if additional_info:
            self.pipeline_status[stage_number]["details"] = additional_info
        
        # Build completion message
        info_parts = []
        if input_count is not None and output_count is not None:
            info_parts.append(f"{input_count:,} → {output_count:,}")
        elif output_count is not None:
            info_parts.append(f"{output_count:,} items")
        
        if additional_info:
            info_parts.append(additional_info)
            
        info_str = " | ".join(info_parts) if info_parts else ""
        timing_str = f"({elapsed:.1f}s)"
        
        message = f"✅ Stage {stage_number} Complete: {stage_name}"
        if info_str:
            message += f" | {info_str}"
        message += f" {timing_str}"
        
        logger.info(message)
        return elapsed
        
    def _skip_stage(self, stage_number: int, reason: str = "Skipped by user") -> None:
        """Mark a stage as skipped."""
        self.pipeline_status[stage_number]["status"] = "⏭️ Skipped"
        self.pipeline_status[stage_number]["details"] = reason

    def _apply_filtering_limits(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply filtering limits to control batch sizes for LLM processing.
        
        Args:
            df: DataFrame with chunks to filter
            
        Returns:
            Sampled DataFrame respecting the filtering limits
        """
        original_count = len(df)
        
        # Apply per-document sampling first if specified
        if self.config.filter_samples_per_doc is not None:
            logger.info(f"Applying per-document sampling: max {self.config.filter_samples_per_doc} chunks per document")
            
            # Group by document and sample chunks within each document
            # Assuming there's a way to identify documents (e.g., through document_url or a doc_id column)
            if "document_url" in df.columns:
                # Safe sampling: use min(requested_samples, available_chunks) per document
                # This prevents the "cannot take a larger sample" error
                df = (df.group_by("document_url")
                      .agg([
                          pl.all().sample(
                              n=pl.min_horizontal(
                                  pl.lit(self.config.filter_samples_per_doc),
                                  pl.col("unique_id").len()
                              )
                          )
                      ])
                      .explode(pl.exclude("document_url")))
                logger.info(f"After per-document sampling: {len(df)} chunks ({original_count - len(df)} removed)")
        
        # Apply total chunk limit if specified
        if self.config.filter_max_total_chunks is not None:
            if len(df) > self.config.filter_max_total_chunks:
                logger.info(f"Applying total chunk limit: sampling {self.config.filter_max_total_chunks} from {len(df)} chunks")
                # Use min() to prevent sampling more than available
                sample_size = min(self.config.filter_max_total_chunks, len(df))
                df = df.sample(sample_size)
                logger.info(f"After total sampling: {len(df)} chunks")
        
        if len(df) != original_count:
            logger.info(f"📊 Filtering limits applied: {original_count} → {len(df)} chunks ({100 * len(df) / original_count:.1f}%)")
        
        return df

    def _smart_read_delta(self, file_path, size_threshold_mb: int = 100) -> pl.DataFrame:
        """Smart Delta file reading based on file size.

        Args:
            file_path: Path to Delta table
            size_threshold_mb: Size threshold in MB for using streaming vs direct read

        Returns:
            DataFrame with loaded data
        """
        try:
            # Get approximate file size by checking directory size
            total_size_bytes = sum(f.stat().st_size for f in file_path.rglob("*") if f.is_file())
            size_mb = total_size_bytes / (1024 * 1024)

            # Convert Path to string for Polars Delta operations
            file_path_str = str(file_path)

            if size_mb > size_threshold_mb:
                logger.info(f"Large file ({size_mb:.1f}MB) - using streaming collection")
                return pl.scan_delta(file_path_str).collect(engine="streaming")
            logger.info(f"Small file ({size_mb:.1f}MB) - using direct read")
            return pl.read_delta(file_path_str)
        except Exception as e:
            logger.warning(f"Could not determine file size, falling back to direct read: {e}")
            return pl.read_delta(str(file_path))

    def load_dataset(self) -> pl.LazyFrame:
        """Load dataset from local parquet file using lazy evaluation.

        Returns:
            Polars LazyFrame with loaded data
        """
        # Don't duplicate logging since process_pipeline already logs this
        # logger.info(f"Loading local dataset from: {self.config.local_data_file}")

        # Check if local data file exists
        if not self.config.local_data_file.exists():
            raise FileNotFoundError(
                f"Local data file not found: {self.config.local_data_file}\n"
                f"Please ensure the processed Natural Questions data is available."
            )

        # Use scan_parquet for lazy loading
        lf = pl.scan_parquet(self.config.local_data_file)

        # Apply sampling if requested
        if self.config.sample_size:
            # Use head/limit for efficient sampling (takes first N records without loading entire dataset)
            lf = lf.head(self.config.sample_size)
            self.state.total_documents = self.config.sample_size
            logger.info(f"  🎯 Limited to first {self.config.sample_size} documents (efficient head/limit operation)")

            # Note: This will create many more chunks after text splitting
            if self.config.sample_chunks:
                logger.info(f"  ✂️ Will further limit to {self.config.sample_chunks} chunks after splitting")
            else:
                estimated_chunks = self.config.sample_size * 50  # Rough estimate: 50 chunks per document
                logger.info(
                    f"  ⚠️ Note: These {self.config.sample_size} documents will likely generate ~{estimated_chunks:,} chunks after text splitting"
                )
        else:
            # For full dataset, stay lazy - count will be done when needed
            self.state.total_documents = None  # Will be set when collected
            if self.config.sample_chunks:
                logger.info(f"  ✂️ Will limit to first {self.config.sample_chunks} chunks after splitting")

        # Mark as complete
        self.state.mark_stage_complete("data_loading", self.config.local_data_file, streaming_mode=True)

        logger.info("  📦 Dataset ready for lazy evaluation")
        return lf

    def validate_data_format(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Validate that the local data has the expected format using lazy evaluation.

        Args:
            lf: Loaded LazyFrame

        Returns:
            LazyFrame with validated format
        """
        # Don't duplicate logging since process_pipeline already logs this
        # logger.info("Validating data format")

        # Check schema without collecting data (avoid performance warning)
        schema = lf.collect_schema()
        required_columns = ["document_url", "question_text", "long_answer_text", "document_html"]
        missing_columns = [col for col in required_columns if col not in schema]

        if missing_columns:
            raise ValueError(
                f"Local data is missing required columns: {missing_columns}\n"
                f"Expected columns: {required_columns}\n"
                f"Found columns: {list(schema.keys())}"
            )

        # Filter out rows with null document_html using lazy operation
        lf_filtered = lf.filter(pl.col("document_html").is_not_null())

        # For logging purposes, we can collect just the count if needed
        # But to stay lazy, we'll defer this until necessary
        self.state.mark_stage_complete("text_extraction", None, streaming_validation=True)

        logger.info("  ✅ Required columns present, filtering applied")
        return lf_filtered

    def chunk_documents(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Chunk documents into smaller pieces for processing using streaming.

        Args:
            lf: LazyFrame with document HTML

        Returns:
            LazyFrame with chunked documents
        """
        # Check if already processed
        output_file = self.config.processed_dir / "documents_chunked.delta"
        if self.state.is_stage_complete("chunking") and output_file.exists():
            logger.info(f"📦 Loading cached chunked documents from {output_file}")
            try:
                # Get file size for informative logging
                total_size_bytes = sum(f.stat().st_size for f in output_file.rglob("*") if f.is_file())
                size_mb = total_size_bytes / (1024 * 1024)
                logger.info(f"  Cache size: {size_mb:.1f}MB")
            except:
                pass
            # LAZY LOADING JUSTIFIED: Returning LazyFrame for further downstream processing
            return pl.scan_delta(str(output_file))

        logger.info("🔄 Generating new document chunks...")

        def generate_unique_id() -> str:
            """Generate unique ID for each chunk."""
            return str(uuid4())

        # For operations requiring UDFs (map_elements), we need to collect in streaming mode
        # then process in batches for memory efficiency

        # First, prepare the lazy frame with basic operations
        lf_prepared = lf.select(["document_html", "document_url"])

        # Process with UDFs (these can't be fully lazy due to Python function calls)
        logger.info("  📄 Extracting text from HTML documents...")

        # Show progress during the intensive processing
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing documents into chunks...", total=None)

            chunked_df = (
                lf_prepared.with_columns(
                    [
                        pl.col("document_html")
                        .map_elements(lambda x: BeautifulSoup(x, "html.parser").get_text(), return_dtype=pl.String)
                        .map_elements(lambda x: self.text_splitter.split_text(x), return_dtype=pl.List(pl.String))
                        .alias("chunked_prompt")
                    ]
                )
                .select(["chunked_prompt", "document_url"])
                .explode("chunked_prompt")
                .with_columns(
                    [
                        pl.col("chunked_prompt").str.len_chars().alias("context_length"),
                        pl.col("chunked_prompt")
                        .map_elements(lambda x: len(self.encoding.encode(x)), return_dtype=pl.Int64)
                        .alias("token_count"),
                        pl.col("chunked_prompt").map_elements(
                            lambda x: re.sub(r"\n{3,}", "\n\n", x), return_dtype=pl.String
                        ),
                        pl.col("chunked_prompt")
                        .map_elements(lambda _: generate_unique_id(), return_dtype=pl.String)
                        .alias("unique_id"),
                    ]
                )
                .filter(pl.col("token_count") <= self.config.max_token_length)
            ).collect(engine="streaming")

        logger.info(f"  ✂️ Generated {len(chunked_df)} chunks from documents")

        # Apply chunk sampling if requested (limits AFTER text splitting)
        if self.config.sample_chunks:
            original_chunk_count = len(chunked_df)
            chunked_df = chunked_df.head(self.config.sample_chunks)
            logger.info(
                f"  🌯 Limited to first {self.config.sample_chunks} chunks out of {original_chunk_count} total chunks"
            )

        # Save chunked documents using Delta for atomic writes and append capabilities
        logger.info(f"  💾 Saving chunks to {output_file}...")
        chunked_df.write_delta(str(output_file), mode="overwrite")

        # Calculate statistics for logging
        stats = chunked_df.select(
            [
                pl.count().alias("count"),
                pl.col("context_length").mean().alias("avg_length"),
                pl.col("token_count").mean().alias("avg_tokens"),
            ]
        ).row(0)

        self.state.mark_stage_complete(
            "chunking",
            output_file,
            chunks_created=stats[0],
            avg_chunk_length=stats[1],
            avg_token_count=stats[2],
            streaming_mode=True,
        )

        logger.info("  📊 Chunk statistics:")
        logger.info(f"     • Total chunks: {stats[0]:,}")
        logger.info(f"     • Avg length: {stats[1]:.0f} chars")
        logger.info(f"     • Avg tokens: {stats[2]:.0f} tokens")

        # Return as lazy frame for downstream processing
        return chunked_df.lazy()

    async def process_pipeline(
        self,
        *,
        skip_embedding: bool = False,
        skip_filtering: bool = False,
        skip_chroma: bool = False,
        use_async_filtering: bool = True,
        shutdown_check: callable = None,
    ) -> dict[str, Any]:
        """Run the complete processing pipeline using streaming where possible.

        Args:
            skip_embedding: Skip embedding generation
            skip_filtering: Skip document filtering
            skip_chroma: Skip ChromaDB ingestion
            use_async_filtering: Use async filtering for better performance
            shutdown_check: Optional function to check for shutdown requests

        Returns:
            Pipeline results and statistics
        """
        # Log pipeline configuration upfront
        logger.info("=" * 60)
        logger.info("📋 PIPELINE PLANNING")
        logger.info("=" * 60)

        # Check what stages are already complete
        existing_embeddings = (self.config.processed_dir / "document_embeddings.delta").exists()
        existing_filtered = (self.config.processed_dir / "filtered_documents.delta").exists()
        existing_chunks = (self.config.processed_dir / "documents_chunked.delta").exists()

        logger.info("🔍 Checking existing data:")
        if existing_chunks:
            logger.info("  ✓ Found chunked documents (will load if needed)")
        else:
            logger.info("  ✗ No chunked documents found (will generate)")

        if existing_embeddings and not skip_embedding:
            logger.info("  ✓ Found existing embeddings (will reuse)")
        elif skip_embedding:
            logger.info("  ⏭️ Embedding generation skipped by user")
        else:
            logger.info("  ✗ No embeddings found (will generate)")

        if existing_filtered and not skip_filtering:
            logger.info("  ℹ️ Found filtered documents (will regenerate for fresh filtering)")
        elif skip_filtering:
            logger.info("  ⏭️ Document filtering skipped by user")
        else:
            logger.info("  ✗ No filtered documents (will filter)")

        # Log what will actually happen
        logger.info("\n📊 EXECUTION PLAN:")
        logger.info("  Stage 1: Data Loading → Will load lazily")
        logger.info("  Stage 2: Data Validation → Will validate schema")
        logger.info(
            "  Stage 3: Document Chunking → {}".format(
                "Will LOAD existing chunks" if existing_chunks else "Will GENERATE new chunks"
            )
        )

        if not skip_embedding:
            if existing_embeddings:
                logger.info("  Stage 4: Embeddings → Will LOAD existing embeddings")
            else:
                logger.info("  Stage 4: Embeddings → Will GENERATE new embeddings")
        else:
            logger.info("  Stage 4: Embeddings → SKIPPED")

        if not skip_filtering:
            logger.info("  Stage 5: LLM Filtering → Will filter documents with LLM")
        else:
            logger.info("  Stage 5: LLM Filtering → SKIPPED")

        if not skip_chroma and not skip_embedding:
            logger.info("  Stage 6: ChromaDB → Will ingest to vector database")
        else:
            logger.info("  Stage 6: ChromaDB → SKIPPED")

        logger.info("=" * 60)
        logger.info("🚀 STARTING EXECUTION")
        logger.info("=" * 60)

        logger.info("Starting WikiText processing pipeline with streaming")

        # Start pipeline timing and resource monitoring
        self.pipeline_start_time = time.time()
        initial_memory, _ = self._get_memory_usage()
        logger.info(f"🚀 Pipeline starting with {initial_memory:.1f}MB memory usage")
        
        # Check for cancellation at pipeline start
        if shutdown_check:
            shutdown_check()

        try:
            results = {"pipeline_config": self.config, "stages": {}, "streaming_enabled": True}
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted before starting")
            raise

        try:
            # Stage 1: Load dataset (lazy)
            if shutdown_check:
                shutdown_check()
            
            description = f"Loading from: {self.config.local_data_file}"
            if self.config.sample_size:
                description += f" | Will take first {self.config.sample_size} documents"
            else:
                description += " | Will load all documents"
            
            self._start_stage(1, "📥 Data Loading", description)
            lf = self.load_dataset()
            self._complete_stage(1, "Data Loading", additional_info="Lazy loaded")
            results["stages"]["data_loading"] = {"status": "completed", "mode": "lazy"}

            # Stage 2: Validate data format (lazy)
            if shutdown_check:
                shutdown_check()
                
            self._start_stage(2, "✅ Data Validation", "Checking for required columns and filtering valid documents...")
            lf_validated = self.validate_data_format(lf)
            self._complete_stage(2, "Data Validation", additional_info="Schema validated")
            results["stages"]["data_validation"] = {"status": "completed", "mode": "lazy"}

            # Stage 3: Chunk documents (uses streaming collection internally)
            if shutdown_check:
                shutdown_check()
                
            description = f"Configuration: chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}"
            if self.config.sample_chunks:
                description += f" | Will limit to first {self.config.sample_chunks} chunks after splitting"
            description += " | Starting text chunking with overlapping windows..."
            
            self._start_stage(3, "✂️ Document Chunking", description)
            chunked_lf = self.chunk_documents(lf_validated)
            # Get chunk count for reporting
            chunk_stats = (
                chunked_lf.select([pl.len().alias("count")])
                .collect()  # No streaming needed for simple count
                .row(0)
            )
            self._complete_stage(3, "Document Chunking", output_count=chunk_stats[0], additional_info="Text chunked with overlaps")
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted during data loading/validation/chunking")
            raise

        # For simple statistics on already-processed data, regular collect is fine
        # (chunked data is typically much smaller than raw input)
        chunk_stats = (
            chunked_lf.select([pl.len().alias("count"), pl.col("context_length").mean().alias("avg_length")])
            .collect()  # No streaming needed for simple aggregations
            .row(0)
        )

        self.state.processed_documents = chunk_stats[0]
        results["stages"]["chunking"] = {
            "chunks_created": chunk_stats[0],
            "avg_chunk_length": chunk_stats[1],
            "status": "completed",
            "mode": "lazy_with_simple_aggregation",
        }

        # Stage 4: Generate embeddings
        if not skip_embedding:
            if shutdown_check:
                shutdown_check()
            try:
                description = f"Model: {self.config.embedding_model} | Collecting chunked documents for embedding generation..."
                self._start_stage(4, "🔮 Embedding Generation", description)

                # STREAMING JUSTIFIED: Collecting potentially large chunked dataset for embedding generation
                # The embedding process needs actual DataFrame, and chunked data can be substantial
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=self.console,
                ) as progress:
                    collect_task = progress.add_task("Collecting chunked documents...", total=None)
                    chunked_df = chunked_lf.collect(engine="streaming")

                logger.info("Starting embedding generation process (this may take several minutes)...")
                embedded_df = self.embedding_generator.load_or_generate_embeddings(chunked_df)
                self._complete_stage(4, "Embedding Generation", input_count=len(chunked_df), output_count=len(embedded_df), additional_info="Embeddings created")
                self.state.embedded_documents = len(embedded_df)
            except KeyboardInterrupt:
                logger.info("Pipeline interrupted during embedding generation")
                raise

            # Validate embeddings
            validation_results = self.embedding_generator.validate_embeddings(embedded_df)

            results["stages"]["embedding_generation"] = {
                "embeddings_generated": len(embedded_df),
                "validation_results": validation_results,
                "status": "completed",
                "mode": "streaming_collection",
            }
            self.state.mark_stage_complete(
                "embedding_generation", self.config.processed_dir / "document_embeddings.delta", **validation_results
            )
        else:
            self._skip_stage(4, "Embedding generation skipped by user")
            embedded_df = None  # Will use chunked_lf directly
            results["stages"]["embedding_generation"] = {"status": "skipped"}

        # Stage 5: Filter documents
        if not skip_filtering:
            if shutdown_check:
                shutdown_check()
            try:
                batch_size = getattr(self.config, "filter_batch_size", 100)
                description = f"Model: {self.config.filter_model} | Batch size: {batch_size} | Approach: Consecutive chunks with question generation"
                self._start_stage(5, "🔍 LLM Filtering", description)

                # STREAMING JUSTIFIED: Large dataset collection for LLM filtering pipeline
                # Only when we haven't already collected for embeddings
                if embedded_df is None:
                    logger.info("Collecting documents for filtering (no embedding data available)...")
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        TimeElapsedColumn(),
                        console=self.console,
                    ) as progress:
                        collect_task = progress.add_task("Collecting documents for filtering...", total=None)
                        chunked_df = chunked_lf.collect(engine="streaming")
                    logger.info(f"Collected {len(chunked_df)} chunks for filtering")
                else:
                    chunked_df = embedded_df
                    logger.info(f"Using {len(chunked_df)} embedded documents for filtering")

                # Apply filtering limits if specified
                sampled_df = self._apply_filtering_limits(chunked_df)
                
                # Use batched filtering with checkpointing for large datasets
                checkpoint_path = Path("logs") / "filtering_checkpoint.json"

                filtered_df, filter_metadata = await self.document_filter.run_filtering_pipeline(
                    sampled_df,
                    use_async=use_async_filtering,
                    batch_size=batch_size,
                    checkpoint_path=str(checkpoint_path),
                )

                self._complete_stage(5, "LLM Filtering", input_count=len(sampled_df), output_count=len(filtered_df), additional_info="Documents filtered by LLM")
            except KeyboardInterrupt:
                logger.info("Pipeline interrupted during LLM filtering")
                raise
            self.state.filtered_documents = len(filtered_df)

            # If we have embeddings, filter them too
            if not skip_embedding and embedded_df is not None:
                embedded_df = embedded_df.filter(
                    pl.col("unique_id").is_in(filtered_df.select("unique_id").to_series().to_list())
                )

            results["stages"]["document_filtering"] = {
                "documents_filtered": len(filtered_df),
                "filter_metadata": filter_metadata,
                "status": "completed",
                "mode": "streaming_collection",
            }
            self.state.mark_stage_complete(
                "document_filtering", self.config.processed_dir / "filtered_documents.delta", **filter_metadata
            )

            # Save filtered documents using Delta for atomic writes
            filtered_output = self.config.processed_dir / "filtered_documents.delta"
            filtered_df.write_delta(str(filtered_output), mode="overwrite")

        else:
            self._skip_stage(5, "LLM filtering skipped by user")
            # STREAMING JUSTIFIED: Only if we need to collect the full dataset
            # and haven't already done so in embedding stage
            if embedded_df is None:
                filtered_df = chunked_lf.collect(engine="streaming")
            else:
                filtered_df = embedded_df
            results["stages"]["document_filtering"] = {"status": "skipped"}

        # Stage 6: Ingest to ChromaDB
        if not skip_chroma and not skip_embedding:
            if shutdown_check:
                shutdown_check()
            try:
                collection_name = f"wikitext_{self.config.dataset_name.replace('/', '_')}"
                description = f"Collection: {collection_name} | Creating/accessing ChromaDB collection..."
                self._start_stage(6, "🗃️ ChromaDB Ingestion", description)
                
                collection = self.chroma_manager.get_or_create_collection(
                    collection_name,
                    metadata={
                        "dataset": self.config.dataset_name,
                        "chunk_size": self.config.chunk_size,
                        "embedding_model": self.config.embedding_model,
                    },
                )

                logger.info(f"Starting ChromaDB ingestion for {len(embedded_df)} documents with embeddings...")
                ingestion_stats = self.chroma_manager.add_documents_in_batches(collection, embedded_df)
                self._complete_stage(6, "ChromaDB Ingestion", input_count=len(embedded_df), output_count=ingestion_stats.get("documents_ingested", len(embedded_df)), additional_info="Vector database updated")
            except KeyboardInterrupt:
                logger.info("Pipeline interrupted during ChromaDB ingestion")
                raise

            results["stages"]["chroma_ingestion"] = {
                "ingestion_stats": ingestion_stats,
                "collection_name": collection_name,
                "status": "completed",
            }
            self.state.mark_stage_complete("chroma_ingestion", None, **ingestion_stats)
        else:
            if skip_chroma:
                self._skip_stage(6, "ChromaDB ingestion skipped by user")
            else:
                self._skip_stage(6, "ChromaDB skipped (no embeddings)")
            results["stages"]["chroma_ingestion"] = {"status": "skipped"}

        # Final results with timing and resource usage
        pipeline_elapsed = time.time() - self.pipeline_start_time
        final_memory, memory_delta = self._get_memory_usage()
        
        logger.info("\n" + "=" * 60)
        logger.info("✨ PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total pipeline time: {pipeline_elapsed:.1f}s")
        logger.info(f"💾 Final memory usage: {final_memory:.1f}MB (+{memory_delta:.1f}MB)")
        
        # Display final status board
        final_status_table = self._create_status_table()
        self.console.print("\n")
        self.console.print(Panel(final_status_table, title="📊 Final Pipeline Status", border_style="green"))
        
        # Add resource summary to results
        results["resource_usage"] = {
            "total_time_seconds": pipeline_elapsed,
            "initial_memory_mb": self.initial_memory,
            "final_memory_mb": final_memory,
            "memory_delta_mb": memory_delta,
            "peak_memory_mb": final_memory  # Could track peak if needed
        }

        results["pipeline_summary"] = self.state.get_progress_summary()

        # Determine final document count based on what stages were run
        if not skip_filtering:
            results["final_document_count"] = len(filtered_df)
        elif not skip_embedding:
            results["final_document_count"] = (
                len(embedded_df) if embedded_df is not None else self.state.processed_documents
            )
        else:
            # Both embedding and filtering skipped, use chunk count from state
            results["final_document_count"] = self.state.processed_documents

        logger.info("🎉 WikiText processing pipeline completed successfully!")
        logger.info(f"Final output: {results['final_document_count']} processed documents")
        logger.info(f"Pipeline summary: {results['pipeline_summary']}")

        return results

    def get_processing_status(self) -> dict[str, Any]:
        """Get current processing status.

        Returns:
            Current pipeline state and progress
        """
        return {
            "config": self.config,
            "state": self.state,
            "progress_summary": self.state.get_progress_summary(),
            "stages": {
                stage.name: {
                    "completed": stage.completed,
                    "output_file": str(stage.output_file) if stage.output_file else None,
                    "metadata": stage.metadata,
                }
                for stage in self.state.stages
            },
        }

    def explain_streaming_plan(self, lf: pl.LazyFrame) -> str:
        """Explain the streaming execution plan for a LazyFrame.

        Args:
            lf: LazyFrame to analyze

        Returns:
            String explanation of the streaming plan
        """
        try:
            return lf.explain(streaming=True)
        except Exception as e:
            return f"Could not generate streaming plan: {e}"

    def demonstrate_streaming_capabilities(self) -> None:
        """Demonstrate various streaming capabilities of the pipeline."""
        logger.info("Demonstrating Polars streaming capabilities")

        # Example 1: Lazy loading with scan_parquet (input data)
        lf = pl.scan_parquet(self.config.local_data_file)
        logger.info("Lazy loaded input data - no memory used yet")

        # Example 2: Check if operations support streaming
        lf_filtered = lf.filter(pl.col("document_html").is_not_null())
        streaming_plan = self.explain_streaming_plan(lf_filtered)
        logger.info(f"Streaming plan for filtering:\n{streaming_plan}")

        # Example 3: Streaming aggregation
        stats_lf = lf_filtered.select(
            [pl.col("document_html").str.len_chars().mean().alias("avg_doc_length"), pl.count().alias("total_docs")]
        )

        # Collect with streaming engine
        stats = stats_lf.collect(engine="streaming")
        logger.info(f"Streaming aggregation results: {stats}")

        # Example 4: Delta operations for intermediate files
        demo_data = lf_filtered.select(["document_url"]).head(100).collect(engine="streaming")
        output_path = self.config.processed_dir / "streaming_demo.delta"
        demo_data.write_delta(str(output_path), mode="overwrite")

        # Example 5: Lazy read from Delta
        demo_lf = pl.scan_delta(str(output_path))
        logger.info(f"Delta streaming plan:\n{self.explain_streaming_plan(demo_lf)}")
        logger.info(f"Data written to {output_path} using Delta with streaming support")

        return

    def resume_from_checkpoint(self, shutdown_check: callable = None) -> dict[str, Any]:
        """Resume processing from the last completed stage.

        Args:
            shutdown_check: Optional function to check for shutdown requests

        Returns:
            Results from resuming pipeline
        """
        logger.info("Resuming pipeline from checkpoint")

        # Find last completed stage
        last_completed = None
        for stage in self.state.stages:
            if stage.completed:
                last_completed = stage

        if last_completed:
            logger.info(f"Resuming from stage: {last_completed.name}")
        else:
            logger.info("No completed stages found, starting from beginning")

        # Run pipeline with appropriate skips
        skip_embedding = self.state.is_stage_complete("embedding_generation")
        skip_filtering = self.state.is_stage_complete("document_filtering")
        skip_chroma = self.state.is_stage_complete("chroma_ingestion")

        return asyncio.run(
            self.process_pipeline(
                skip_embedding=skip_embedding,
                skip_filtering=skip_filtering,
                skip_chroma=skip_chroma,
                shutdown_check=shutdown_check,
            )
        )
