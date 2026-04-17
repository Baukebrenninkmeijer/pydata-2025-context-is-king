#!/usr/bin/env python3
"""
Simplified WikiText Query Generation Pipeline v2.0

A clear, linear pipeline with explicit stage boundaries, transparent file I/O,
and simple checkpoint/recovery mechanism.

Usage:
    # Run full pipeline
    python generate_wikitext_queries_v2.py

    # Resume from a specific stage
    python generate_wikitext_queries_v2.py --start-from stage4

    # Skip certain stages
    python generate_wikitext_queries_v2.py --skip-stages 5,6

    # Quick test with small sample
    python generate_wikitext_queries_v2.py --sample-size 100 --filter-max-total-chunks 50

    # Dry run to see what would happen
    python generate_wikitext_queries_v2.py --dry-run
"""

import argparse
import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.console import Group

# Import from parent context_is_king module
from context_is_king.query_generation.config import WikiTextConfig
from context_is_king.query_generation.simple_pipeline import SimplifiedPipeline, STAGES

console = Console()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the simplified pipeline."""
    parser = argparse.ArgumentParser(
        description="WikiText Query Generation Pipeline v2.0 - Simplified and Clear",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run full pipeline
  %(prog)s --start-from stage3                # Resume from stage 3
  %(prog)s --skip-stages 5,6                  # Skip filtering and ChromaDB
  %(prog)s --sample-size 100                  # Quick test with 100 docs
  %(prog)s --dry-run                          # See what would happen
  %(prog)s --filter-samples-per-doc 3         # Limit LLM filtering
  %(prog)s --start-from stage4 --verbose      # Resume with detailed logs
  %(prog)s --force-reingest --sample-size 50  # Fresh start with small sample
        """,
    )

    # Basic configuration
    basic = parser.add_argument_group("Basic Configuration")
    basic.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/processed/nq_question_answer.parquet"),
        help="Input data file (default: data/processed/nq_question_answer.parquet)",
    )
    basic.add_argument("--sample-size", type=int, help="Number of documents to process (for testing)")
    basic.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Data directory for outputs (default: data/)"
    )

    # Pipeline Control
    control = parser.add_argument_group("Pipeline Control")
    control.add_argument(
        "--start-from",
        choices=["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"],
        help="Start pipeline from specific stage",
    )
    control.add_argument("--skip-stages", type=str, help="Skip specific stages (comma-separated: 1,2,5)")
    control.add_argument("--dry-run", action="store_true", help="Show what would happen without executing")
    control.add_argument(
        "--force-reingest",
        action="store_true",
        help="Force re-ingestion by clearing existing ChromaDB collections and cached data",
    )

    # Text Processing
    text = parser.add_argument_group("Text Processing")
    text.add_argument("--chunk-size", type=int, default=1900, help="Chunk size (default: 1900)")
    text.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)")
    text.add_argument("--sample-chunks", type=int, help="Limit total chunks after splitting")

    # Models
    models = parser.add_argument_group("Model Configuration")
    models.add_argument(
        "--embedding-model",
        default="azure/text-embedding-3-small",
        help="Embedding model (default: azure/text-embedding-3-small)",
    )
    models.add_argument(
        "--filter-model", default="azure/gpt-5-nano", help="LLM filtering model (default: azure/gpt-5-nano)"
    )

    # Filtering Controls
    filtering = parser.add_argument_group("LLM Filtering Controls")
    filtering.add_argument("--filter-samples-per-doc", type=int, help="Max chunks per document for LLM filtering")
    filtering.add_argument("--filter-max-total-chunks", type=int, help="Max total chunks for LLM filtering")
    filtering.add_argument(
        "--filter-batch-size",
        type=int,
        default=20,
        help="Batch size for LLM filtering (default: 20, reduced to prevent timeouts)",
    )

    # System
    system = parser.add_argument_group("System Configuration")
    system.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    system.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def show_pipeline_plan(args: argparse.Namespace) -> None:
    """Show what the pipeline would do without executing with rich visuals."""

    # Beautiful header
    header = Text()
    header.append("🔍 ", style="bold blue")
    header.append("WikiText Pipeline v2.0", style="bold white")
    header.append(" - ", style="dim white")
    header.append("Dry Run", style="bold yellow")

    console.print()
    console.print(
        Panel(
            Align.center(header),
            style="bold blue",
            border_style="blue",
            width=85,
            title="[bold white on blue] PREVIEW MODE [/bold white on blue]",
            title_align="center",
        )
    )

    # Determine starting stage and skipped stages
    start_stage = 1
    if args.start_from:
        start_stage = int(args.start_from[-1])

    skip_stages = []
    if args.skip_stages:
        skip_stages = [int(s.strip()) for s in args.skip_stages.split(",")]

    # Configuration table
    config_table = Table.grid(padding=(0, 1))
    config_table.add_column(style="dim cyan", no_wrap=True, min_width=18)
    config_table.add_column(style="white")

    config_table.add_row("📂 Data file:", f"[green]{args.data_file}[/green]")
    config_table.add_row("📏 Sample size:", f"[yellow]{args.sample_size or 'Full dataset'}[/yellow]")
    config_table.add_row("🔮 Embedding model:", f"[blue]{args.embedding_model}[/blue]")
    config_table.add_row("🔍 Filter model:", f"[blue]{args.filter_model}[/blue]")
    config_table.add_row("🚀 Start from:", f"[cyan]Stage {start_stage}[/cyan]")
    if skip_stages:
        config_table.add_row("⏭️  Skip stages:", f"[red]{skip_stages}[/red]")

    # Show configuration
    console.print()
    console.print(
        Panel(
            Group(Text("📋 Configuration", style="bold cyan"), "", config_table),
            border_style="cyan",
            padding=(1, 2),
            width=85,
        )
    )

    # Stages execution plan
    stages_table = Table.grid(padding=(0, 1))
    stages_table.add_column(style="white", no_wrap=True, min_width=15)
    stages_table.add_column(style="white")
    stages_table.add_column(style="dim white")

    for stage_num in range(1, 7):
        emoji = STAGES[stage_num]["emoji"]
        name = STAGES[stage_num]["name"]

        if stage_num < start_stage:
            status = "[dim]Will be skipped (before start)[/dim]"
            stage_text = f"[dim]{emoji} Stage {stage_num}[/dim]"
            name_text = f"[dim]{name}[/dim]"
        elif stage_num in skip_stages:
            status = "[red]SKIPPED by user[/red]"
            stage_text = f"[dim]{emoji} Stage {stage_num}[/dim]"
            name_text = f"[dim]{name}[/dim]"
        else:
            status = "[green]✅ Will execute[/green]"
            stage_text = f"[bold white]{emoji} Stage {stage_num}[/bold white]"
            name_text = f"[white]{name}[/white]"

        stages_table.add_row(stage_text, name_text, status)

    console.print()
    console.print(
        Panel(
            Group(Text("🚀 Execution Plan", style="bold cyan"), "", stages_table),
            border_style="cyan",
            padding=(1, 2),
            width=85,
        )
    )

    # Files that would be created
    files_table = Table.grid(padding=(0, 1))
    files_table.add_column(style="yellow", no_wrap=True, min_width=5)
    files_table.add_column(style="white")

    data_dir = args.data_dir / "processed"
    if start_stage <= 3 and 3 not in skip_stages:
        files_table.add_row("📄", f"[green]{data_dir}/documents_chunked.parquet[/green]")
    if start_stage <= 4 and 4 not in skip_stages:
        files_table.add_row("📄", f"[green]{data_dir}/document_embeddings.parquet[/green]")
    if start_stage <= 5 and 5 not in skip_stages:
        files_table.add_row("📄", f"[green]{data_dir}/filtered_documents.parquet[/green]")
    if start_stage <= 6 and 6 not in skip_stages:
        files_table.add_row("🗃️", f"[green]ChromaDB collection in {args.data_dir}/vector_stores/chroma_db/[/green]")

    if files_table.row_count > 0:
        console.print()
        console.print(
            Panel(
                Group(Text("📁 Files to be Created", style="bold cyan"), "", files_table),
                border_style="cyan",
                padding=(1, 2),
                width=85,
            )
        )

    # Execution command
    cmd_text = f"python {Path(__file__).name} {' '.join(sys.argv[1:]).replace('--dry-run', '').strip()}"
    console.print()
    console.print(
        Panel(
            Group(Text("💡 To Execute", style="bold yellow"), "", Text(cmd_text, style="bold white")),
            border_style="yellow",
            padding=(1, 2),
            width=85,
        )
    )


def create_config(args: argparse.Namespace) -> WikiTextConfig:
    """Create configuration from arguments."""
    return WikiTextConfig(
        # Paths
        data_dir=args.data_dir,
        processed_dir=args.data_dir / "processed",
        chroma_db_path=args.data_dir / "vector_stores" / "chroma_db",
        local_data_file=args.data_file,
        # Sampling
        sample_size=args.sample_size,
        sample_chunks=args.sample_chunks,
        # Text processing
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_token_length=2000,
        # Models
        embedding_model=args.embedding_model,
        filter_model=args.filter_model,
        # Filtering controls
        filter_samples_per_doc=args.filter_samples_per_doc,
        filter_max_total_chunks=args.filter_max_total_chunks,
        filter_batch_size=args.filter_batch_size,
        # System
        log_level=args.log_level if args.verbose else "INFO",
    )


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration matching the SimplifiedPipeline setup."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Remove existing handlers to avoid duplicates
    logger.remove()

    # Add file handler - same config as SimplifiedPipeline
    logger.add(
        log_dir / "simple_pipeline.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
    )

    # Optionally add console handler for errors/warnings
    logger.add(
        sys.stderr,
        level="ERROR",
        format="<red>{time:HH:mm:ss}</red> | <level>{level}</level> | {message}",
        colorize=True,
    )


def main() -> None:
    """Main entry point for simplified pipeline with beautiful visuals."""

    try:
        args = parse_arguments()

        # Setup logging early
        setup_logging(args.log_level if args.verbose else "INFO")

        # Handle dry run
        if args.dry_run:
            show_pipeline_plan(args)
            return

        # Create beautiful header

        # Main title
        title = Text()
        title.append("🚀 ", style="bold blue")
        title.append("WikiText Query Generation Pipeline", style="bold white")
        title.append(" v2.0", style="bold cyan")

        # Subtitle
        subtitle = Text()
        subtitle.append("Simplified", style="green")
        subtitle.append(" • ", style="dim white")
        subtitle.append("Clear", style="yellow")
        subtitle.append(" • ", style="dim white")
        subtitle.append("Transparent", style="magenta")

        console.print()
        console.print(
            Panel(
                Group(Align.center(title), "", Align.center(subtitle)),
                style="bold blue",
                border_style="bright_blue",
                width=85,
                title="[bold white on blue] WIKITEXT PIPELINE [/bold white on blue]",
                title_align="center",
                padding=(1, 2),
            )
        )

        # Create configuration
        config = create_config(args)

        # Show startup configuration
        startup_table = Table.grid(padding=(0, 1))
        startup_table.add_column(style="dim cyan", no_wrap=True, min_width=15)
        startup_table.add_column(style="white")

        startup_table.add_row("📂 Data file:", f"[green]{config.local_data_file}[/green]")
        if config.sample_size:
            startup_table.add_row("📏 Sample size:", f"[yellow]{config.sample_size:,} documents[/yellow]")
        startup_table.add_row("🔮 Embedding model:", f"[blue]{config.embedding_model}[/blue]")
        startup_table.add_row("🔍 Filter model:", f"[blue]{config.filter_model}[/blue]")

        console.print()
        console.print(
            Panel(
                Group(Text("🎛️  Pipeline Configuration", style="bold cyan"), "", startup_table),
                border_style="cyan",
                padding=(1, 2),
                width=85,
            )
        )

        # Initialize pipeline
        pipeline = SimplifiedPipeline(config)

        # Handle force-reingest by clearing existing data
        if args.force_reingest:
            console.print()
            console.print(
                Panel(
                    Group(
                        Text("🔄 Force Re-ingestion Requested", style="bold yellow"),
                        "",
                        Text("Clearing existing data...", style="white"),
                    ),
                    border_style="yellow",
                    width=85,
                    title="[bold white on yellow] CLEARING DATA [/bold white on yellow]",
                    title_align="center",
                    padding=(1, 2),
                )
            )

            # Clear ChromaDB collections
            collections = pipeline.chroma_manager.list_collections()
            for collection_info in collections:
                collection_name = collection_info.get("name")
                if collection_name:
                    pipeline.chroma_manager.delete_collection(collection_name)
                    console.print(f"[yellow]  ✓ Cleared collection: {collection_name}[/yellow]")

            # Clear cached files and checkpoints
            import shutil

            cache_dirs = [
                config.processed_dir / "embeddings",
                config.processed_dir / "filtered",
                config.processed_dir / "chunked",
                config.data_dir / ".pipeline",  # Clear checkpoint directory
            ]

            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    console.print(f"[yellow]  ✓ Cleared cache: {cache_dir}[/yellow]")

            # Clear individual processed files
            processed_files = [
                "stage1_loaded_documents.parquet",
                "stage2_cleaned_documents.parquet",
                "documents_chunked.parquet",
                "filtered_documents.parquet",
            ]

            for filename in processed_files:
                file_path = config.processed_dir / filename
                if file_path.exists():
                    file_path.unlink()
                    console.print(f"[yellow]  ✓ Cleared file: {filename}[/yellow]")

            console.print()
            console.print(
                Panel(
                    "✅ Data cleared successfully - starting fresh pipeline",
                    border_style="green",
                    width=85,
                    title="[bold white on green] CLEANUP COMPLETE [/bold white on green]",
                    title_align="center",
                )
            )

        # Determine starting stage and show resume info
        start_stage = 1
        if args.start_from:
            start_stage = int(args.start_from[-1])
            console.print()
            console.print(
                Panel(
                    f"🔄 [bold yellow]RESUMING FROM STAGE {start_stage}[/bold yellow]",
                    style="yellow",
                    border_style="yellow",
                    width=85,
                    title_align="center",
                )
            )

        # Determine skipped stages and show skip info
        skip_stages = []
        if args.skip_stages:
            skip_stages = [int(s.strip()) for s in args.skip_stages.split(",")]
            skip_names = [STAGES[s]["name"] for s in skip_stages]
            console.print()
            console.print(
                Panel(
                    f"⏭️  [bold red]SKIPPING STAGES:[/bold red] [white]{', '.join(skip_names)}[/white]",
                    style="red",
                    border_style="red",
                    width=85,
                    title_align="center",
                )
            )

        # Run pipeline
        results = pipeline.run_from_stage(start_stage=start_stage, skip_stages=skip_stages)

        return results

    except KeyboardInterrupt:
        console.print()
        console.print(
            Panel(
                Group(
                    Text("⚠️  Pipeline Interrupted", style="bold yellow"),
                    "",
                    Text("Progress has been saved automatically.", style="white"),
                    Text("Use --start-from to resume from any completed stage.", style="dim white"),
                ),
                style="yellow",
                border_style="yellow",
                width=85,
                title="[bold white on yellow] INTERRUPTED [/bold white on yellow]",
                title_align="center",
                padding=(1, 2),
            )
        )
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        import traceback

        # Log the error first - this is the key fix!
        error_type = type(e).__name__
        error_msg = str(e)

        # Log error with full traceback to file
        logger.error(f"Pipeline failed with {error_type}: {error_msg}")
        logger.exception("Full exception details:")

        # Get last few lines of traceback for CLI display
        tb_lines = traceback.format_exc().strip().split("\n")
        # Get the last 3-4 relevant lines (skip the generic "Traceback" line)
        context_lines = tb_lines[-4:-1] if len(tb_lines) > 4 else tb_lines[1:-1]
        context_text = "\n".join(context_lines)

        error_text = Group(
            Text(f"❌ {error_type}: {error_msg}", style="bold red"),
            "",
            Text("Error Context:", style="yellow"),
            Text(context_text, style="red"),
            "",
            Text("💡 Full details logged to logs/simple_pipeline.log", style="dim cyan"),
        )

        if args.verbose if "args" in locals() else False:
            # Show full traceback in verbose mode
            error_text = Group(
                Text(f"❌ {error_type}: {error_msg}", style="bold red"),
                "",
                Text("Full Traceback:", style="dim yellow"),
                Text(traceback.format_exc(), style="red"),
            )

        console.print()
        console.print(
            Panel(
                error_text,
                style="red",
                border_style="bright_red",
                width=85,
                title="[bold white on red] ERROR [/bold white on red]",
                title_align="center",
                padding=(1, 2),
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
