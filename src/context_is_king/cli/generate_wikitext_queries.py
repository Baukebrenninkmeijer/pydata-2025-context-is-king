#!/usr/bin/env python3
"""
WikiText Query Generation Script with Polars Streaming Support

Process WikiText datasets to create synthetic query datasets for research experiments.
Uses Polars streaming engine for memory-efficient processing of large datasets.
Supports the full pipeline: data loading, text extraction, chunking, embedding generation,
LLM-based filtering, and ChromaDB ingestion.

Features:
    - Smart streaming: Only uses streaming when beneficial (file size > 100MB)
    - Lazy evaluation with scan_parquet for minimal memory usage
    - Delta Lake for atomic incremental processing
    - Checkpoint/recovery system for long-running pipelines
    - Semaphore-based rate limiting for API calls
    - Graceful shutdown: Proper signal handling with async task cancellation

Usage examples:
    # Full pipeline with local data (uses streaming)
    python scripts/generate_wikitext_queries.py --local-data-file data/processed/nq_question_answer.parquet --sample-size 1000

    # Skip certain stages
    python scripts/generate_wikitext_queries.py --skip-embedding --skip-filtering

    # Resume from checkpoint
    python scripts/generate_wikitext_queries.py --resume

    # Custom configuration with adjusted concurrency
    python scripts/generate_wikitext_queries.py --chunk-size 2000 --embedding-batch-size 50 --max-concurrent-requests 50

    # Demonstrate streaming capabilities
    python scripts/generate_wikitext_queries.py --demo-streaming
"""

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

# Initialize rich console
console = Console()

# Import from parent context_is_king module
from context_is_king.query_generation import WikiTextConfig, WikiTextProcessor

# Global variables for graceful shutdown
_shutdown_requested = False
_main_task = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested, _main_task
    _shutdown_requested = True
    console.print(f"\n[yellow]⚠️  Received signal {signum} - initiating graceful shutdown...[/yellow]")
    console.print("[dim]Saving progress and cleaning up...[/dim]")

    # Cancel the main task if it exists
    if _main_task and not _main_task.done():
        _main_task.cancel()


def setup_signals():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def is_shutdown_requested() -> bool:
    """Check if graceful shutdown has been requested."""
    return _shutdown_requested


def check_shutdown() -> None:
    """Check for shutdown request and raise CancelledError if requested."""
    if _shutdown_requested:
        console.print("[yellow]Shutdown requested - stopping execution[/yellow]")
        raise asyncio.CancelledError("Shutdown requested by signal handler")


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration - file + console bridge for important messages."""
    logger.remove()  # Remove default handler

    # Add file handler for all logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "wikitext_generation.log",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="1 week",
    )

    # Add console handler for all stage-related messages
    def console_filter(record):
        """Filter to show stage messages and important progress on console."""
        message = record["message"]
        level = record["level"].name

        # Always show ERROR and WARNING messages
        if level in ["ERROR", "WARNING"]:
            return True

        # Show all messages that contain stage indicators
        stage_indicators = [
            "STAGE",
            "PIPELINE",
            "EXECUTION",
            "Stage",
            "=" * 40,  # Section separators
            "=" * 50,  # Stage separators
            "=" * 60,  # Main separators
        ]

        # Show all messages with stage emojis (these mark important sections)
        stage_emojis = [
            "📥",  # Stage 1: Data loading
            "✅",  # Stage 2: Validation
            "✂️",  # Stage 3: Chunking
            "🔮",  # Stage 4: Embeddings
            "🔍",  # Stage 5: Filtering
            "🗃️",  # Stage 6: ChromaDB
            "📋",  # Planning
            "📊",  # Configuration/stats
            "🚀",  # Starting/execution
            "✨",  # Complete
            "🎉",  # Success
            "⚡",  # Processing
            "💡",  # Tips/info
            "⚠️",  # Warnings
            "🔄",  # Loading/processing
            "💾",  # Saving/checkpoints
        ]

        # Show progress and completion messages
        progress_keywords = [
            "Complete:",
            "completed",
            "successfully",
            "Starting",
            "Processing",
            "Generating",
            "Collecting",
            "Loading",
            "Saving",
            "chunks for",
            "documents",
            "embeddings",
            "ChromaDB",
            "Batch",
            "progress:",
            "Will load",
            "Will generate",
            "Will filter",
            "Will ingest",
            "SKIPPED",
            "Cache",
            "Found",
        ]

        # Check all filter categories
        return (
            any(indicator in message for indicator in stage_indicators)
            or any(emoji in message for emoji in stage_emojis)
            or any(keyword in message for keyword in progress_keywords)
            or level == "INFO"  # Show all INFO level messages for now - we can tune this later
        )

    def format_log_message(msg) -> None:
        """Format log messages with improved stage indicators and consistent styling."""
        message = msg.rstrip()

        # Stage headers (highest priority - bold and prominent)
        if any(stage in message for stage in ["STAGE 1", "STAGE 2", "STAGE 3", "STAGE 4", "STAGE 5", "STAGE 6"]):
            console.print(f"[bold blue]{message}[/bold blue]")
        elif "PIPELINE" in message or "EXECUTION" in message:
            console.print(f"[bold magenta]{message}[/bold magenta]")

        # Section separators
        elif "=" * 40 in message or "=" * 50 in message or "=" * 60 in message:
            console.print(f"[dim cyan]{message}[/dim cyan]")

        # Completion and success messages
        elif any(success in message for success in ["Complete:", "completed successfully", "🎉"]):
            console.print(f"[bold green]{message}[/bold green]")
        elif "✅" in message and "Stage" in message:
            console.print(f"[green]{message}[/green]")

        # Progress and processing messages
        elif any(process in message for process in ["Starting", "Processing", "Generating", "Collecting"]):
            console.print(f"[yellow]{message}[/yellow]")
        elif "Batch" in message and ("completed" in message or "progress:" in message):
            console.print(f"[green]  ✓ {message}[/green]")

        # Cache and loading messages
        elif any(cache in message for cache in ["Loading cached", "Found cached", "💾", "Loading from"]):
            console.print(f"[cyan]{message}[/cyan]")
        elif any(load in message for load in ["Loading", "Saving", "Cache"]):
            console.print(f"[blue]{message}[/blue]")

        # Configuration and planning
        elif any(
            config in message
            for config in ["Configuration:", "Will load", "Will generate", "Will filter", "Will ingest"]
        ):
            console.print(f"[cyan]{message}[/cyan]")
        elif "SKIPPED" in message:
            console.print(f"[yellow]{message}[/yellow]")

        # Statistics and counts
        elif any(stat in message for stat in ["chunks", "documents", "embeddings", "Total", "statistics:"]):
            console.print(f"[blue]{message}[/blue]")

        # Warnings and info
        elif any(warn in message for warn in ["⚠️", "Note:", "Warning"]):
            console.print(f"[yellow]{message}[/yellow]")
        elif any(info in message for info in ["ℹ️", "💡", "Tip:"]):
            console.print(f"[cyan]{message}[/cyan]")

        # Error conditions (should not normally reach here due to filter, but safety net)
        elif any(error in message for error in ["Error", "Failed", "Exception"]):
            console.print(f"[red]{message}[/red]")

        # Default formatting for other messages
        else:
            console.print(f"[white]{message}[/white]")

    logger.add(format_log_message, level="INFO", filter=console_filter, format="{message}")


def create_header_panel(config: WikiTextConfig) -> Panel:
    """Create the header panel with configuration info."""
    return Panel.fit(
        f"[bold blue]🔧 WikiText Query Generation Pipeline[/bold blue]\n"
        f"[dim]Data File:[/dim] [cyan]{config.local_data_file}[/cyan]\n"
        f"[dim]Chunk Size:[/dim] [green]{config.chunk_size}[/green] | "
        f"[dim]Overlap:[/dim] [green]{config.chunk_overlap}[/green]\n"
        f"[dim]Embedding Model:[/dim] [yellow]{config.embedding_model}[/yellow]\n"
        f"[dim]Filter Model:[/dim] [yellow]{config.filter_model}[/yellow]\n"
        f"[dim]Concurrency:[/dim] [magenta]{config.max_concurrent_requests}[/magenta] | "
        f"[dim]Batch Size:[/dim] [magenta]{config.filter_batch_size}[/magenta]\n"
        f"[dim]Filtering Limits:[/dim] "
        f"[cyan]{config.filter_samples_per_doc or 'unlimited'}[/cyan] per doc | "
        f"[cyan]{config.filter_max_total_chunks or 'unlimited'}[/cyan] total",
        title="📊 Pipeline Configuration",
        border_style="blue",
    )


def create_status_table(results: dict) -> Table:
    """Create a status table from pipeline results."""
    table = Table(title="🚀 Pipeline Progress", show_header=True, header_style="bold magenta")
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    stages = results.get("stages", {})

    for stage_name, stage_info in stages.items():
        status = stage_info.get("status", "unknown")

        # Format status with emoji
        status_display = {
            "completed": "✅ Completed",
            "skipped": "⏩ Skipped",
            "failed": "❌ Failed",
            "in_progress": "⏳ In Progress",
        }.get(status, f"❓ {status}")

        # Extract relevant details
        details = []
        if "documents_filtered" in stage_info:
            details.append(f"{stage_info['documents_filtered']} docs")
        if "embeddings_generated" in stage_info:
            details.append(f"{stage_info['embeddings_generated']} embeddings")
        if "documents_ingested" in stage_info:
            details.append(f"{stage_info['documents_ingested']} ingested")
        if "mode" in stage_info:
            details.append(f"mode: {stage_info['mode']}")

        detail_str = " | ".join(details) if details else "—"

        table.add_row(stage_name.replace("_", " ").title(), status_display, detail_str)

    return table


def display_collections_table(collections: list) -> None:
    """Display ChromaDB collections in a nice table."""
    table = Table(title="🗃️ ChromaDB Collections", show_header=True, header_style="bold magenta")
    table.add_column("Collection Name", style="cyan", no_wrap=True)
    table.add_column("Document Count", style="green", justify="right")
    table.add_column("Status", style="yellow")

    for collection in collections:
        name = collection.get("name", "Unknown")
        count = collection.get("document_count", 0)
        error = collection.get("error")

        if error:
            table.add_row(name, "—", f"❌ {error}")
        else:
            table.add_row(name, f"{count:,}", "✅ Active")

    console.print(table)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate query datasets from WikiText data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --local-data-file data/processed/nq_question_answer.parquet --sample-size 1000
  %(prog)s --skip-embedding --skip-filtering --chunk-size 2000
  %(prog)s --resume --log-level DEBUG
  %(prog)s --filter-model azure/gpt-4o-mini --max-concurrent 25
  %(prog)s --filter-samples-per-doc 5 --filter-max-total-chunks 1000
  %(prog)s --filter-samples-per-doc 3 --skip-embedding --skip-chroma
  %(prog)s --force-reingest --sample-size 100
        """,
    )

    # Dataset configuration
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--local-data-file",
        type=Path,
        default=Path("data/processed/nq_question_answer.parquet"),
        help="Path to local processed data file (default: data/processed/nq_question_answer.parquet)",
    )
    dataset_group.add_argument(
        "--sample-size",
        type=int,
        help="Number of documents to sample BEFORE chunking (default: full dataset). Note: 1 document creates many chunks!",
    )
    dataset_group.add_argument(
        "--sample-chunks",
        type=int,
        help="Number of chunks to keep AFTER text splitting (default: all chunks). Use this for quick testing.",
    )

    # Text processing
    text_group = parser.add_argument_group("Text Processing")
    text_group.add_argument(
        "--chunk-size", type=int, default=1900, help="Chunk size for text splitting (default: 1900)"
    )
    text_group.add_argument(
        "--chunk-overlap", type=int, default=200, help="Chunk overlap for text splitting (default: 200)"
    )
    text_group.add_argument(
        "--max-token-length", type=int, default=2000, help="Maximum token length for chunks (default: 2000)"
    )

    # Embedding configuration
    embedding_group = parser.add_argument_group("Embedding Configuration")
    embedding_group.add_argument(
        "--embedding-model",
        default="azure/text-embedding-3-small",
        help="Embedding model name (default: azure/text-embedding-3-small)",
    )
    embedding_group.add_argument(
        "--embedding-batch-size", type=int, default=300, help="Batch size for embedding generation (default: 100)"
    )
    embedding_group.add_argument(
        "--embedding-max-batch-size",
        type=int,
        default=50000,
        help="Maximum batch size for memory management (default: 50000)",
    )

    # Filtering configuration
    filter_group = parser.add_argument_group("Document Filtering")
    filter_group.add_argument(
        "--filter-model", default="azure/gpt-5-nano", help="Model for document filtering (default: azure/gpt-5-nano)"
    )
    filter_group.add_argument(
        "--max-concurrent", type=int, default=50, help="Maximum concurrent requests for filtering (default: 50)"
    )
    filter_group.add_argument(
        "--requests-per-minute", type=int, default=500, help="Rate limit: requests per minute (default: 500)"
    )
    filter_group.add_argument(
        "--tokens-per-minute", type=int, default=300000, help="Rate limit: tokens per minute (default: 300000)"
    )
    filter_group.add_argument(
        "--sync-filtering", action="store_true", help="Use synchronous filtering instead of async"
    )
    filter_group.add_argument(
        "--filter-samples-per-doc",
        type=int,
        help="Maximum number of chunks per document to send for LLM filtering (default: no limit)",
    )
    filter_group.add_argument(
        "--filter-max-total-chunks",
        type=int,
        help="Maximum total number of chunks to send to LLM for filtering (default: no limit)",
    )

    # Pipeline control
    pipeline_group = parser.add_argument_group("Pipeline Control")
    pipeline_group.add_argument("--skip-embedding", action="store_true", help="Skip embedding generation stage")
    pipeline_group.add_argument("--skip-filtering", action="store_true", help="Skip document filtering stage")
    pipeline_group.add_argument("--skip-chroma", action="store_true", help="Skip ChromaDB ingestion stage")
    pipeline_group.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    pipeline_group.add_argument(
        "--force-reingest",
        action="store_true",
        help="Force re-ingestion by clearing existing ChromaDB collections and regenerating all data",
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory (default: data/)")
    output_group.add_argument("--output-file", type=Path, help="Output file for results (default: auto-generated)")

    # System configuration
    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    system_group.add_argument("--orq-api-key", help="ORQ API key (default: from ORQ_API_KEY env var)")

    # Status and utilities
    util_group = parser.add_argument_group("Utilities")
    util_group.add_argument("--status", action="store_true", help="Show current processing status and exit")
    util_group.add_argument("--list-collections", action="store_true", help="List ChromaDB collections and exit")
    util_group.add_argument(
        "--demo-streaming", action="store_true", help="Demonstrate Polars streaming capabilities and exit"
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> WikiTextConfig:
    """Create WikiTextConfig from command line arguments."""

    # Override ORQ API key if provided
    import os

    if args.orq_api_key:
        os.environ["ORQ_API_KEY"] = args.orq_api_key

    config = WikiTextConfig(
        # Data paths
        data_dir=args.data_dir,
        processed_dir=args.data_dir / "processed",
        chroma_db_path=args.data_dir / "vector_stores" / "chroma_db",
        # Dataset
        local_data_file=args.local_data_file,
        sample_size=args.sample_size,
        sample_chunks=args.sample_chunks,
        # Text processing
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_token_length=args.max_token_length,
        # Embeddings
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        # Filtering
        filter_model=args.filter_model,
        max_concurrent_requests=args.max_concurrent,
        requests_per_minute=args.requests_per_minute,
        tokens_per_minute=args.tokens_per_minute,
        filter_samples_per_doc=args.filter_samples_per_doc,
        filter_max_total_chunks=args.filter_max_total_chunks,
        # System
        log_level=args.log_level,
    )

    return config


def save_results(results: dict[str, Any], output_file: Path) -> None:
    """Save pipeline results to file."""

    # Convert non-serializable objects
    serializable_results = {}
    for key, value in results.items():
        if key == "pipeline_config":
            # Convert WikiTextConfig to dict
            serializable_results[key] = {
                attr: getattr(value, attr)
                for attr in dir(value)
                if not attr.startswith("_") and not callable(getattr(value, attr))
            }
        else:
            serializable_results[key] = value

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_file}")


async def main_pipeline(args: argparse.Namespace) -> None:
    """Main pipeline execution function with rich terminal interface."""
    setup_logging(args.log_level)

    try:
        # Initialize configuration with status
        with Status("[bold green]Initializing configuration...", console=console):
            config = create_config_from_args(args)

        # Display header panel
        console.print(create_header_panel(config))

        # Initialize processor with status indicator
        with Status("[bold yellow]Initializing WikiText processor (ChromaDB connection)...", console=console):
            processor = WikiTextProcessor(config)

        console.print("[green]✅ WikiText processor initialized successfully![/green]")

        # Handle force-reingest by clearing existing data
        if args.force_reingest:
            console.print("[bold yellow]🔄 Force re-ingestion requested[/bold yellow]")

            # Get information about what will be cleared
            collections = processor.chroma_manager.list_collections()
            collection_names = [c.get("name") for c in collections if c.get("name")]

            import shutil

            cache_dirs = [
                config.processed_dir / "embeddings",
                config.processed_dir / "filtered",
                config.processed_dir / "chunked",
            ]
            existing_cache_dirs = [d for d in cache_dirs if d.exists()]

            # Show what will be deleted
            console.print("[bold red]⚠️  This will permanently delete:[/bold red]")
            if collection_names:
                console.print(f"  • {len(collection_names)} ChromaDB collection(s): {', '.join(collection_names)}")
            if existing_cache_dirs:
                console.print(
                    f"  • {len(existing_cache_dirs)} cache director(ies): {', '.join(str(d) for d in existing_cache_dirs)}"
                )
            console.print("  • All checkpoint data")

            # Ask for confirmation
            confirm = input("\n🔴 Are you sure you want to proceed with force re-ingestion? [y/N]: ").strip().lower()
            if confirm not in ["y", "yes"]:
                console.print("[yellow]Force re-ingestion cancelled[/yellow]")
                return

            console.print("[bold yellow]Clearing existing data...[/bold yellow]")

            with Status("[yellow]Clearing ChromaDB collections and cached embeddings...", console=console):
                # Clear ChromaDB collections
                for collection_name in collection_names:
                    processor.chroma_manager.delete_collection(collection_name)
                    console.print(f"[yellow]  ✓ Cleared collection: {collection_name}[/yellow]")

                # Clear cached embeddings and processed files
                for cache_dir in existing_cache_dirs:
                    shutil.rmtree(cache_dir)
                    console.print(f"[yellow]  ✓ Cleared cache directory: {cache_dir}[/yellow]")

                # Clear checkpoint state
                if hasattr(processor, "state") and hasattr(processor.state, "checkpoint_file"):
                    checkpoint_file = processor.state.checkpoint_file
                    if checkpoint_file and checkpoint_file.exists():
                        checkpoint_file.unlink()
                        console.print(f"[yellow]  ✓ Cleared checkpoint: {checkpoint_file}[/yellow]")

            console.print("[green]✅ Data cleared successfully - proceeding with full re-ingestion[/green]")

            # Force all stages to run (override skip flags when force-reingest is used)
            console.print("[dim]Note: Force-reingest overrides skip flags to ensure complete re-processing[/dim]")
            args.skip_embedding = False
            args.skip_filtering = False
            args.skip_chroma = False

        # Handle utility commands with rich output
        if args.status:
            console.print("\n")
            with Status("[cyan]Retrieving processing status...", console=console):
                status = processor.get_processing_status()

            console.print(create_status_table(status))
            console.print("\n[dim]Full status saved to logs/wikitext_generation.log[/dim]")
            return

        if args.list_collections:
            console.print("\n")
            with Status("[cyan]Loading ChromaDB collections...", console=console):
                collections = processor.chroma_manager.list_collections()

            display_collections_table(collections)
            return

        if args.demo_streaming:
            console.print("\n")
            console.print(
                Panel.fit(
                    "[bold blue]🌊 Polars Streaming Capabilities Demo[/bold blue]\n\n"
                    "[yellow]This will demonstrate streaming processing capabilities[/yellow]\n"
                    "[dim]Check logs/wikitext_generation.log for detailed output...[/dim]",
                    title="🚀 Streaming Demo",
                    border_style="cyan",
                )
            )

            with Status("[bold cyan]Running streaming demonstration...", console=console):
                processor.demonstrate_streaming_capabilities()

            console.print("[green]✅ Streaming demonstration completed![/green]")
            return

        # Run main pipeline with detailed progress tracking
        console.print("\n[bold magenta]🚀 Starting WikiText Pipeline[/bold magenta]")

        if args.resume:
            with Status("[yellow]Resuming pipeline from checkpoint...", console=console):
                # Create a periodic shutdown checker task for resume as well
                async def periodic_shutdown_check():
                    while not _shutdown_requested:
                        await asyncio.sleep(1.0)  # Check every second
                        check_shutdown()  # This will raise CancelledError if shutdown requested

                shutdown_task = asyncio.create_task(periodic_shutdown_check())

                try:
                    results = processor.resume_from_checkpoint(shutdown_check=check_shutdown)
                finally:
                    # Clean up the shutdown checker task
                    shutdown_task.cancel()
                    try:
                        await shutdown_task
                    except asyncio.CancelledError:
                        pass
        else:
            # Show pipeline overview with live status tracking
            console.print("[cyan]Starting WikiText processing pipeline with live progress tracking...[/cyan]\n")

            # Show what stages will run
            stage_info = []
            stage_info.append("📥 Stage 1: Data Loading (lazy)")
            stage_info.append("✅ Stage 2: Data Validation")
            stage_info.append("✂️ Stage 3: Document Chunking")
            if not args.skip_embedding:
                stage_info.append(f"🔮 Stage 4: Embedding Generation ({config.embedding_model})")
            if not args.skip_filtering:
                stage_info.append(
                    f"🔍 Stage 5: LLM Filtering ({config.filter_model}) [yellow]⏳ Long-running operation[/yellow]"
                )
            if not args.skip_chroma:
                stage_info.append("🗃️ Stage 6: ChromaDB Ingestion")

            for info in stage_info:
                console.print(f"  {info}")

            console.print("\n[bold cyan]📊 Configuration:[/bold cyan]")
            console.print(
                f"  • Batch size: [yellow]{config.filter_batch_size}[/yellow] docs | "
                f"Concurrency: [yellow]{config.max_concurrent_requests}[/yellow] | "
                f"Rate limit: [yellow]{config.tokens_per_minute:,}[/yellow] tokens/min"
            )

            console.print("\n[bold green]🚀 Starting pipeline execution...[/bold green]")
            console.print("[dim]💡 Stage progress will appear below in real-time[/dim]\n")

            # Add ongoing status for long operations
            if not args.skip_filtering:
                console.print(
                    "[yellow]⚠️  Note: Stage 5 (LLM Filtering) may take several minutes "
                    "depending on dataset size[/yellow]"
                )
                console.print("[dim]    You'll see batch-by-batch progress updates during filtering[/dim]\n")

            # Show current stage indicator
            with Status("[bold blue]🔄 Initializing pipeline stages...", console=console):
                await asyncio.sleep(0.5)  # Brief initialization delay for visual feedback

            # Create a periodic shutdown checker task
            async def periodic_shutdown_check():
                while not _shutdown_requested:
                    await asyncio.sleep(1.0)  # Check every second
                    check_shutdown()  # This will raise CancelledError if shutdown requested

            shutdown_task = asyncio.create_task(periodic_shutdown_check())

            try:
                results = await processor.process_pipeline(
                    skip_embedding=args.skip_embedding,
                    skip_filtering=args.skip_filtering,
                    skip_chroma=args.skip_chroma,
                    use_async_filtering=not args.sync_filtering,
                    shutdown_check=check_shutdown,
                )
            finally:
                # Clean up the shutdown checker task
                shutdown_task.cancel()
                try:
                    await shutdown_task
                except asyncio.CancelledError:
                    pass

        # Save results with status
        console.print("\n")
        with Status("[green]Saving pipeline results...", console=console):
            if args.output_file:
                output_file = args.output_file
            else:
                timestamp = processor.state.stages[0].metadata.get("timestamp", "unknown")
                output_file = config.processed_dir / f"pipeline_results_{timestamp}.json"
            save_results(results, output_file)

        # Display final results summary
        console.print("\n")
        console.print(create_status_table(results))

        # Create success panel
        summary = results.get("pipeline_summary", {})

        # Format timing information more prominently
        total_time = summary.get("total_execution_time", 0)
        if total_time:
            time_display = f"[bold green]⏱️  {total_time:.1f}s[/bold green]"
        else:
            time_display = "[bold green]⏱️  Completed[/bold green]"

        success_panel = Panel.fit(
            f"[bold green]🎉 Pipeline Completed Successfully![/bold green]\n\n"
            f"[bright_white]Final Results:[/bright_white]\n"
            f"[bright_cyan]• Documents Processed:[/bright_cyan] "
            f"[bold yellow]{results.get('final_document_count', 0):,}[/bold yellow]\n"
            f"[bright_cyan]• Stages Completed:[/bright_cyan] "
            f"[bold yellow]{summary.get('stages_completed', 0)}/{summary.get('total_stages', 0)}[/bold yellow] "
            f"[bright_green]({summary.get('progress_percent', 0):.1f}%)[/bright_green]\n"
            f"[bright_cyan]• Filtered Documents:[/bright_cyan] "
            f"[bold yellow]{summary.get('filtered_documents', 0):,}[/bold yellow]\n"
            f"[bright_cyan]• Embedded Documents:[/bright_cyan] "
            f"[bold yellow]{summary.get('embedded_documents', 0):,}[/bold yellow]\n"
            f"[bright_cyan]• Execution Time:[/bright_cyan] {time_display}\n\n"
            f"[bright_white]Results saved to:[/bright_white] "
            f"[bold green]{output_file}[/bold green]",
            title="✨ Pipeline Complete",
            border_style="bright_green",
        )
        console.print(success_panel)

    except asyncio.CancelledError:
        console.print("\n")
        console.print(
            Panel.fit(
                "[bold yellow]⚠️ Pipeline Interrupted[/bold yellow]\n\n"
                "[cyan]Progress has been saved.[/cyan]\n"
                "[dim]Use --resume to continue from checkpoint.[/dim]",
                title="🛑 Interrupted",
                border_style="yellow",
            )
        )
        raise  # Re-raise to let main() handle the exit

    except Exception as e:
        console.print("\n")
        console.print(
            Panel.fit(
                f"[bold red]❌ Pipeline Failed[/bold red]\n\n"
                f"[yellow]Error:[/yellow] [red]{e}[/red]\n\n"
                f"[dim]Check logs/wikitext_generation.log for detailed error information.[/dim]",
                title="🚨 Error",
                border_style="red",
            )
        )
        raise  # Re-raise to let main() handle the exit


async def main_async(args: argparse.Namespace) -> None:
    """Async main entry point with proper signal handling and cancellation support."""
    global _main_task

    setup_signals()  # Setup signal handlers for graceful cancellation

    try:
        # Create the main pipeline task
        _main_task = asyncio.create_task(main_pipeline(args))

        # Wait for the task to complete
        await _main_task

    except asyncio.CancelledError:
        console.print("\n[yellow]🛑 Pipeline cancelled by user request[/yellow]")
        console.print("[dim]Final cleanup and exit...[/dim]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Pipeline interrupted by KeyboardInterrupt[/yellow]")
        if _main_task and not _main_task.done():
            _main_task.cancel()
            try:
                # Wait up to 10 seconds for graceful shutdown
                await asyncio.wait_for(_main_task, timeout=10.0)
            except asyncio.CancelledError:
                console.print("[green]✅ Pipeline cancelled gracefully[/green]")
            except asyncio.TimeoutError:
                console.print("[red]⚠️  Graceful shutdown timed out, forcing exit[/red]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        sys.exit(1)


def main() -> None:
    """Synchronous main entry point that handles argument parsing."""
    # Parse arguments first (this handles --help and --version without entering async context)
    args = parse_arguments()
    # Only enter async context if we're actually running the pipeline
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
