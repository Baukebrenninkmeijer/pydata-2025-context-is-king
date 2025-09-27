"""
Simplified WikiText Query Generation Pipeline

A clear, linear pipeline with explicit stage boundaries, transparent file I/O,
and simple checkpoint/recovery mechanism.

Design Principles:
- Linear, synchronous main flow (async only where necessary)
- Clear stage boundaries with explicit inputs/outputs
- Transparent file I/O reporting
- Progress bars for long operations
- Simple checkpoint system
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
import tiktoken
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .chroma import ChromaManager
from .config import WikiTextConfig
from .embeddings import EmbeddingGenerator
from .filters import DocumentFilter

# Initialize console
console = Console()

# Stage configuration
STAGES = {
    1: {"name": "Data Loading", "emoji": "📥"},
    2: {"name": "Data Validation & Cleaning", "emoji": "✅"},
    3: {"name": "Document Chunking", "emoji": "✂️"},
    4: {"name": "Embedding Generation", "emoji": "🔮"},
    5: {"name": "LLM Filtering", "emoji": "🔍"},
    6: {"name": "ChromaDB Ingestion", "emoji": "🗃️"},
}


class PipelineCheckpoint:
    """Simple checkpoint management for pipeline stages."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, stage: int, output_file: Path | None = None, stats: dict | None = None) -> None:
        """Save checkpoint after stage completion."""
        # Ensure checkpoint directory exists (in case it was cleared by --force-reingest)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "stage": stage,
            "stage_name": STAGES[stage]["name"],
            "completed_at": datetime.now().isoformat(),
            "output_file": str(output_file) if output_file else None,
            "stats": stats or {},
            "next_stage": stage + 1 if stage < 6 else None,
        }

        checkpoint_file = self.checkpoint_dir / f"stage_{stage}_complete.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_file}")

    def load_checkpoint(self, stage: int) -> dict | None:
        """Load checkpoint for a specific stage."""
        checkpoint_file = self.checkpoint_dir / f"stage_{stage}_complete.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                return json.load(f)
        return None

    def get_last_completed_stage(self) -> int | None:
        """Find the highest completed stage."""
        for stage in reversed(range(1, 7)):
            if self.load_checkpoint(stage):
                return stage
        return None

    def is_stage_complete(self, stage: int) -> bool:
        """Check if a stage is complete."""
        return self.load_checkpoint(stage) is not None


def print_stage_header(
    stage_num: int,
    input_file: str | None = None,
    output_file: str | None = None,
    action: str = "",
    current_stage: int = None,
    total_stages: int = 6,
) -> None:
    """Print a beautiful stage header with rich visuals and I/O information."""
    stage_info = STAGES[stage_num]
    emoji = stage_info["emoji"]
    name = stage_info["name"]

    # Rich components are now imported at module level

    # Main header with optional progress
    if current_stage and total_stages:
        progress_text = f" [dim]({current_stage}/{total_stages})[/dim]"
    else:
        progress_text = ""

    header_text = (
        f"{emoji} [bold white]STAGE {stage_num}[/bold white]: [bold cyan]{name.upper()}[/bold cyan]{progress_text}"
    )

    # Create info table
    info_table = Table.grid(padding=(0, 1))
    info_table.add_column(style="dim cyan", no_wrap=True)
    info_table.add_column(style="white")

    if input_file:
        info_table.add_row("📂 Input:", f"[green]{input_file}[/green]")
    if output_file:
        info_table.add_row("📝 Output:", f"[yellow]{output_file}[/yellow]")
    if action:
        # Handle multi-line actions
        action_lines = action.split("\n")
        for i, line in enumerate(action_lines):
            if i == 0:
                info_table.add_row("📋 Action:", f"[white]{line}[/white]")
            else:
                info_table.add_row("", f"[dim white]{line}[/dim white]")

    # Combine header and table using Group
    content = Group(Text.from_markup(header_text), "", info_table)

    # Print with styled panel
    console.print()
    console.print(
        Panel(
            content,
            style="bright_blue",
            border_style="blue",
            padding=(1, 2),
            width=85,
            title=f"[bold white]⚡ STAGE {stage_num}[/bold white]",
            title_align="left",
        )
    )
    console.print()


def print_stage_complete(
    stage_num: int, stats: dict[str, Any], saved_to: str | None = None, next_stage: int | None = None
) -> None:
    """Print a beautiful stage completion summary with rich visuals."""

    # Rich components already imported at module level

    # Create completion table
    completion_table = Table.grid(padding=(0, 1))
    completion_table.add_column(style="dim green", no_wrap=True, min_width=20)
    completion_table.add_column(style="white")

    # Add stats to table
    for key, value in stats.items():
        # Skip certain internal keys
        if key in ["status", "query_instructions"]:
            continue

        # Format the key nicely
        display_key = key.replace("_", " ").title() + ":"

        # Format the value with colors and formatting
        if isinstance(value, (int, float)):
            if key in ["elapsed_time", "time", "duration"]:
                formatted_value = f"[cyan]{value:.1f}s[/cyan]"
            elif isinstance(value, int) and value > 1000:
                formatted_value = f"[yellow]{value:,}[/yellow]"
            elif isinstance(value, float):
                formatted_value = f"[yellow]{value:.2f}[/yellow]"
            else:
                formatted_value = f"[yellow]{value}[/yellow]"
        elif isinstance(value, str) and "%" in str(value):
            formatted_value = f"[magenta]{value}[/magenta]"
        elif isinstance(value, bool):
            formatted_value = f"[{'green' if value else 'red'}]{value}[/{'green' if value else 'red'}]"
        else:
            formatted_value = f"[white]{value}[/white]"

        completion_table.add_row(f"  {display_key}", formatted_value)

    # Build the content
    content_parts = []
    content_parts.append(f"[bold green]✅ STAGE {stage_num} COMPLETED[/bold green]")
    content_parts.append("")
    content_parts.append(completion_table)

    if saved_to:
        content_parts.append("")
        content_parts.append(f"[dim green]💾 Saved to:[/dim green] [bold yellow]{saved_to}[/bold yellow]")

    if next_stage and next_stage <= 6:
        content_parts.append("")
        content_parts.append(
            f"[dim cyan]📌 To resume:[/dim cyan] [bold white]--start-from stage{next_stage}[/bold white]"
        )

    # Join content
    # Group already imported
    content = Group(*[Text.from_markup(part) if isinstance(part, str) else part for part in content_parts])

    console.print(
        Panel(
            content,
            style="bright_green",
            border_style="green",
            padding=(1, 2),
            width=85,
            title=f"[bold white]✅ STAGE {stage_num} COMPLETE[/bold white]",
            title_align="left",
        )
    )
    console.print()


def print_pipeline_complete(total_time: float, final_stats: dict[str, Any]) -> None:
    """Print a spectacular pipeline completion summary with rich visuals."""

    # Rich components already imported at module level
    # Group already imported
    # Align already imported

    # Create a beautiful completion banner
    banner_text = Text()
    banner_text.append("🎉 ", style="bold yellow")
    banner_text.append("PIPELINE COMPLETE", style="bold green")
    banner_text.append(" 🎉", style="bold yellow")

    # Time formatting
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    if minutes > 0:
        time_text = f"⏱️  [bold green]{minutes}m {seconds}s[/bold green]"
    else:
        time_text = f"⏱️  [bold green]{seconds}s[/bold green]"

    # Create stats table
    stats_table = Table.grid(padding=(0, 2))
    stats_table.add_column(style="dim cyan", no_wrap=True)
    stats_table.add_column(style="bold white")

    for key, value in final_stats.items():
        display_key = f"📊 {key.replace('_', ' ').title()}:"
        if isinstance(value, int):
            formatted_value = f"[yellow]{value:,}[/yellow]"
        else:
            formatted_value = f"[white]{value}[/white]"
        stats_table.add_row(display_key, formatted_value)

    # Create celebration elements
    celebration = "🎊 " * 20

    # Combine all elements
    content = Group(
        Align.center(banner_text),
        "",
        Align.center(Text(time_text, style="bold")),
        "",
        stats_table,
        "",
        Align.center(Text(celebration, style="dim yellow")),
    )

    console.print()
    console.print(
        Panel(
            content,
            style="bold green",
            border_style="bright_green",
            padding=(2, 3),
            width=85,
            title="[bold white on green] SUCCESS [/bold white on green]",
            title_align="center",
        )
    )
    console.print()


class SimplifiedPipeline:
    """Simplified WikiText processing pipeline with clear stages and transparent I/O."""

    def __init__(self, config: WikiTextConfig):
        self.config = config
        self.checkpoint = PipelineCheckpoint(config.data_dir / ".pipeline")

        # Initialize components (lazy initialization)
        self._embedding_generator = None
        self._document_filter = None
        self._chroma_manager = None

        # Initialize text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4o")
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Setup logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        logger.configure(
            handlers=[
                {
                    "sink": log_dir / "simple_pipeline.log",
                    "level": config.log_level,
                    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                    "rotation": "10 MB",
                }
            ]
        )

    @property
    def embedding_generator(self) -> EmbeddingGenerator:
        """Lazy initialization of embedding generator."""
        if self._embedding_generator is None:
            self._embedding_generator = EmbeddingGenerator(self.config)
        return self._embedding_generator

    @property
    def document_filter(self) -> DocumentFilter:
        """Lazy initialization of document filter."""
        if self._document_filter is None:
            self._document_filter = DocumentFilter(self.config)
        return self._document_filter

    @property
    def chroma_manager(self) -> ChromaManager:
        """Lazy initialization of chroma manager."""
        if self._chroma_manager is None:
            self._chroma_manager = ChromaManager(self.config)
        return self._chroma_manager

    def run_from_stage(self, start_stage: int = 1, skip_stages: list[int] = None) -> dict[str, Any]:
        """Run pipeline starting from a specific stage with beautiful progress visualization."""
        skip_stages = skip_stages or []
        pipeline_start_time = time.time()

        # Rich components already imported at module level

        # Create pipeline overview table
        overview_table = Table.grid(padding=(0, 2))
        overview_table.add_column(style="white", no_wrap=True, min_width=15)
        overview_table.add_column(style="white")
        overview_table.add_column(style="dim white")

        # Populate overview table
        for stage_num in range(1, 7):
            emoji = STAGES[stage_num]["emoji"]
            name = STAGES[stage_num]["name"]

            if stage_num < start_stage:
                status = "[dim]Skipped (before start)[/dim]"
                stage_text = f"[dim]{emoji} Stage {stage_num}[/dim]"
                name_text = f"[dim]{name}[/dim]"
            elif stage_num in skip_stages:
                status = "[red]Will skip[/red]"
                stage_text = f"[dim]{emoji} Stage {stage_num}[/dim]"
                name_text = f"[dim]{name}[/dim]"
            elif self.checkpoint.is_stage_complete(stage_num):
                status = "[green]Already complete ✓[/green]"
                stage_text = f"[green]{emoji} Stage {stage_num}[/green]"
                name_text = f"[green]{name}[/green]"
            else:
                status = "[cyan]Will execute[/cyan]"
                stage_text = f"[bold white]{emoji} Stage {stage_num}[/bold white]"
                name_text = f"[white]{name}[/white]"

            overview_table.add_row(stage_text, name_text, status)

        console.print()
        console.print(
            Panel(
                Group(Align.center(Text("🎯 Pipeline Overview", style="bold white")), "", overview_table),
                style="bold blue",
                border_style="blue",
                padding=(1, 2),
                width=85,
                title="[bold white on blue] EXECUTION PLAN [/bold white on blue]",
                title_align="center",
            )
        )

        # Execute stages
        results = {}
        current_data = None
        executed_stages = 0

        # Calculate total stages to execute
        total_stages_to_execute = sum(
            1 for s in range(start_stage, 7) if s not in skip_stages and not self.checkpoint.is_stage_complete(s)
        )

        for stage in range(start_stage, 7):
            if stage in skip_stages:
                # Show skip message with style
                console.print()
                console.print(
                    Panel(
                        f"⏭️  [bold yellow]SKIPPING STAGE {stage}:[/bold yellow] [white]{STAGES[stage]['name']}[/white]",
                        style="yellow",
                        border_style="yellow",
                        width=85,
                        title_align="center",
                    )
                )
                continue

            # Check if stage is already complete
            if self.checkpoint.is_stage_complete(stage):
                console.print()
                console.print(
                    Panel(
                        f"✅ [bold green]STAGE {stage} ALREADY COMPLETE[/bold green]\n"
                        f"[white]Loading from checkpoint...[/white]",
                        style="green",
                        border_style="green",
                        width=85,
                        title_align="center",
                    )
                )

                checkpoint = self.checkpoint.load_checkpoint(stage)
                if checkpoint and checkpoint.get("output_file"):
                    output_file = Path(checkpoint["output_file"])
                    if output_file.exists():
                        current_data = pl.read_parquet(output_file)
                        console.print(f"[green]📂 Loaded: {output_file} ({len(current_data):,} rows)[/green]")
                continue

            # Execute stage with visual separation
            executed_stages += 1
            stage_method = getattr(self, f"_run_stage_{stage}")
            current_data, stage_results = stage_method(current_data, executed_stages, total_stages_to_execute)
            results[f"stage_{stage}"] = stage_results

        # Calculate final results
        total_time = time.time() - pipeline_start_time
        final_stats = {
            "Total processing time": f"{total_time // 60:.0f}m {total_time % 60:.0f}s",
            "Stages completed": len([k for k in results if results[k].get("status") == "completed"]),
            "Final document count": len(current_data) if current_data is not None else 0,
        }

        print_pipeline_complete(total_time, final_stats)

        return {"results": results, "final_stats": final_stats, "total_time": total_time, "final_data": current_data}

    def _run_stage_1(
        self, input_data: Any = None, current_stage: int = 1, total_stages: int = 6
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Stage 1: Data Loading"""
        stage_start_time = time.time()

        input_file = str(self.config.local_data_file)
        sample_text = f" (sampling {self.config.sample_size:,} documents)" if self.config.sample_size else ""
        action = f"Load documents from parquet file{sample_text}"

        print_stage_header(
            1,
            input_file=input_file,
            output_file="In-memory DataFrame",
            action=action,
            current_stage=current_stage,
            total_stages=total_stages,
        )

        # Load data with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Loading documents..."),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Loading", total=100)

            # Load the parquet file
            try:
                df = pl.read_parquet(input_file)
                progress.update(task, advance=50)

                # Apply sampling if specified
                original_count = len(df)
                if self.config.sample_size and self.config.sample_size < original_count:
                    df = df.head(self.config.sample_size)
                    progress.update(task, advance=30)

                # Basic validation
                required_columns = ["document_html", "document_url"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")

                progress.update(task, advance=20)

            except Exception as e:
                logger.error(f"Failed to load data: {e}")
                raise RuntimeError(f"Stage 1 failed: {e}") from e

        # Calculate statistics
        final_count = len(df)
        stage_time = time.time() - stage_start_time

        stats = {
            "documents_loaded": final_count,
            "original_count": original_count,
            "sampling_applied": bool(self.config.sample_size and self.config.sample_size < original_count),
            "file_size_mb": round(Path(input_file).stat().st_size / (1024 * 1024), 1),
            "elapsed_time": stage_time,
        }

        # Save intermediate data for checkpoint recovery
        temp_output_file = self.config.processed_dir / "stage1_loaded_documents.parquet"
        temp_output_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(temp_output_file)

        # Print completion
        print_stage_complete(1, stats, saved_to=str(temp_output_file), next_stage=2)

        # Save checkpoint with output file
        self.checkpoint.save_checkpoint(1, output_file=temp_output_file, stats=stats)

        logger.info(f"Stage 1 completed: {final_count:,} documents loaded in {stage_time:.1f}s")

        return df, {"status": "completed", **stats}

    def _run_stage_2(
        self, input_data: pl.DataFrame, current_stage: int = 1, total_stages: int = 6
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Stage 2: Data Validation & Cleaning"""
        stage_start_time = time.time()

        # Handle resumption from checkpoint - load raw documents if needed
        if input_data is None:
            stage1_file = self.config.processed_dir / "stage1_loaded_documents.parquet"
            if not stage1_file.exists():
                logger.error(f"Stage 2 cannot resume: stage1_loaded_documents.parquet not found at {stage1_file}")
                raise RuntimeError("Stage 2 cannot resume: stage1_loaded_documents.parquet not found")
            input_data = pl.read_parquet(stage1_file)
            logger.info(f"Loaded {len(input_data):,} raw documents from {stage1_file}")

        original_count = len(input_data)
        action = "Validate schema, clean HTML, and filter invalid documents"

        print_stage_header(
            2,
            input_file=f"In-memory DataFrame ({original_count:,} documents)",
            output_file="In-memory cleaned DataFrame",
            action=action,
            current_stage=current_stage,
            total_stages=total_stages,
        )

        # Validation and cleaning with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Task 1: Schema validation
            validate_task = progress.add_task("Validating schema...", total=100)

            # Check required columns
            required_columns = ["document_html", "document_url"]
            optional_columns = ["document_title", "document_id"]

            missing_required = [col for col in required_columns if col not in input_data.columns]
            if missing_required:
                logger.error(f"Stage 2 validation failed: Missing required columns: {missing_required}")
                logger.error(f"Available columns: {input_data.columns}")
                raise ValueError(f"Missing required columns: {missing_required}")

            progress.update(validate_task, advance=100)

            # Task 2: Clean and filter data
            clean_task = progress.add_task("Cleaning and filtering documents...", total=original_count)

            # Filter out documents with null HTML
            df_filtered = input_data.filter(pl.col("document_html").is_not_null())
            progress.update(clean_task, advance=original_count // 4)

            # Filter out empty HTML
            df_filtered = df_filtered.filter(pl.col("document_html").str.len_chars() > 50)
            progress.update(clean_task, advance=original_count // 4)

            # Filter out documents with null URLs
            df_filtered = df_filtered.filter(pl.col("document_url").is_not_null())
            progress.update(clean_task, advance=original_count // 4)

            # Add document length statistics
            df_filtered = df_filtered.with_columns(
                [
                    pl.col("document_html").str.len_chars().alias("html_length"),
                    pl.col("document_url").str.len_chars().alias("url_length"),
                ]
            )
            progress.update(clean_task, advance=original_count // 4)

        # Calculate cleaning statistics
        final_count = len(df_filtered)
        null_html_removed = original_count - len(input_data.filter(pl.col("document_html").is_not_null()))
        short_html_removed = len(df_filtered.filter(pl.col("html_length") <= 50))
        null_url_removed = len(input_data.filter(pl.col("document_url").is_not_null())) - len(
            df_filtered.filter(pl.col("document_url").is_not_null())
        )

        stage_time = time.time() - stage_start_time

        stats = {
            "documents_input": original_count,
            "documents_output": final_count,
            "documents_removed": original_count - final_count,
            "null_html_removed": null_html_removed,
            "short_html_removed": short_html_removed,
            "null_url_removed": null_url_removed,
            "pass_rate": f"{100 * final_count / original_count:.1f}%",
            "avg_html_length": int(df_filtered.select(pl.col("html_length").mean()).item()),
            "elapsed_time": stage_time,
        }

        # Save intermediate data for checkpoint recovery
        temp_output_file = self.config.processed_dir / "stage2_cleaned_documents.parquet"
        temp_output_file.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.write_parquet(temp_output_file)

        # Print completion
        print_stage_complete(2, stats, saved_to=str(temp_output_file), next_stage=3)

        # Save checkpoint with output file
        self.checkpoint.save_checkpoint(2, output_file=temp_output_file, stats=stats)

        logger.info(
            f"Stage 2 completed: {final_count:,} valid documents ({original_count - final_count:,} removed) in {stage_time:.1f}s"
        )

        return df_filtered, {"status": "completed", **stats}

    def _run_stage_3(
        self, input_data: pl.DataFrame, current_stage: int = 1, total_stages: int = 6
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Stage 3: Document Chunking"""
        stage_start_time = time.time()

        # Handle resumption from checkpoint - load cleaned documents if needed
        if input_data is None:
            stage2_file = self.config.processed_dir / "stage2_cleaned_documents.parquet"
            if not stage2_file.exists():
                logger.error(f"Stage 3 cannot resume: stage2_cleaned_documents.parquet not found at {stage2_file}")
                raise RuntimeError("Stage 3 cannot resume: stage2_cleaned_documents.parquet not found")
            input_data = pl.read_parquet(stage2_file)
            logger.info(f"Loaded {len(input_data):,} cleaned documents from {stage2_file}")

        input_count = len(input_data)
        output_file = self.config.processed_dir / "documents_chunked.parquet"

        # Add chunk sampling info to action
        sample_text = f" (limiting to {self.config.sample_chunks:,} chunks)" if self.config.sample_chunks else ""
        action = f"Extract text from HTML and split into overlapping chunks{sample_text}"

        print_stage_header(
            3,
            input_file=f"In-memory DataFrame ({input_count:,} documents)",
            output_file=str(output_file),
            action=action,
            current_stage=current_stage,
            total_stages=total_stages,
        )

        chunks_data = []

        # Process documents with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Processing documents..."),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            doc_task = progress.add_task("Extracting text and chunking...", total=input_count)

            for i, row in enumerate(input_data.iter_rows(named=True)):
                try:
                    # Extract text from HTML
                    soup = BeautifulSoup(row["document_html"], "html.parser")
                    text = soup.get_text()

                    # Clean up text
                    text = re.sub(r"\n\s*\n", "\n\n", text)  # Normalize whitespace
                    text = text.strip()

                    if len(text) < 100:  # Skip very short documents
                        continue

                    # Split into chunks
                    text_chunks = self.text_splitter.split_text(text)

                    # Create chunk records
                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        # Calculate token count
                        token_count = len(self.encoding.encode(chunk_text, disallowed_special=()))

                        # Skip chunks that are too long
                        if token_count > self.config.max_token_length:
                            continue

                        # Create chunk data
                        chunk_data = {
                            "unique_id": str(uuid4()),
                            "document_url": row["document_url"],
                            "chunk_index": chunk_idx,
                            "chunked_prompt": chunk_text,
                            "context_length": len(chunk_text),
                            "token_count": token_count,
                            "html_length": row.get("html_length", 0),
                        }

                        # Add optional fields if they exist
                        if "document_title" in row:
                            chunk_data["document_title"] = row["document_title"]
                        if "document_id" in row:
                            chunk_data["document_id"] = row["document_id"]

                        chunks_data.append(chunk_data)

                except Exception as e:
                    logger.warning(f"Failed to process document {i}: {e}")
                    continue

                progress.update(doc_task, advance=1)

                # Memory management for large datasets
                if len(chunks_data) % 10000 == 0 and len(chunks_data) > 0:
                    logger.info(f"Processed {i + 1:,} documents, generated {len(chunks_data):,} chunks")

        # Create chunks DataFrame
        if not chunks_data:
            logger.error(f"Stage 3 failed: No valid chunks were generated from {input_count} documents")
            logger.error("Possible causes: All documents too short, processing errors, or invalid HTML content")
            raise RuntimeError("No valid chunks were generated")

        chunks_df = pl.DataFrame(chunks_data)

        # Apply chunk sampling if specified
        original_chunk_count = len(chunks_df)
        if self.config.sample_chunks and self.config.sample_chunks < original_chunk_count:
            chunks_df = chunks_df.head(self.config.sample_chunks)
            logger.info(f"Limited to first {self.config.sample_chunks:,} chunks out of {original_chunk_count:,} total")

        # Save to parquet
        with Progress(SpinnerColumn(), TextColumn("[bold blue]Saving chunks to file..."), console=console) as progress:
            save_task = progress.add_task("Saving", total=100)

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save chunks
            chunks_df.write_parquet(output_file)
            progress.update(save_task, advance=100)

        # Calculate statistics
        final_chunk_count = len(chunks_df)
        avg_chunk_length = chunks_df.select(pl.col("context_length").mean()).item()
        avg_token_count = chunks_df.select(pl.col("token_count").mean()).item()

        stage_time = time.time() - stage_start_time

        stats = {
            "documents_processed": input_count,
            "chunks_created": final_chunk_count,
            "original_chunks": original_chunk_count,
            "chunks_per_document": round(final_chunk_count / input_count, 1),
            "avg_chunk_length": int(avg_chunk_length),
            "avg_token_count": int(avg_token_count),
            "sampling_applied": bool(self.config.sample_chunks and self.config.sample_chunks < original_chunk_count),
            "elapsed_time": stage_time,
        }

        # Print completion
        print_stage_complete(3, stats, saved_to=str(output_file), next_stage=4)

        # Save checkpoint
        self.checkpoint.save_checkpoint(3, output_file=output_file, stats=stats)

        logger.info(
            f"Stage 3 completed: {final_chunk_count:,} chunks from {input_count:,} documents in {stage_time:.1f}s"
        )

        return chunks_df, {"status": "completed", **stats}

    def _run_stage_4(
        self, input_data: pl.DataFrame, current_stage: int = 1, total_stages: int = 6
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Stage 4: Embedding Generation & ChromaDB Storage"""
        stage_start_time = time.time()

        # Handle resumption from checkpoint - load chunks if needed
        if input_data is None:
            chunks_file = self.config.processed_dir / "documents_chunked.parquet"
            if not chunks_file.exists():
                logger.error(f"Stage 4 cannot resume: documents_chunked.parquet not found at {chunks_file}")
                raise RuntimeError("Stage 4 cannot resume: documents_chunked.parquet not found")
            input_data = pl.read_parquet(chunks_file)
            logger.info(f"Loaded {len(input_data):,} chunks from {chunks_file}")

        input_count = len(input_data)
        collection_name = f"wikitext_{self.config.dataset_name.replace('/', '_').replace('-', '_')}"
        chroma_path = self.config.chroma_db_path / collection_name

        action = f"Generate embeddings using {self.config.embedding_model} ({self.config.embedding_dimensions}D)"
        batch_info = f"Batch size: {self.config.embedding_batch_size}, Checkpoint every: {self.config.embedding_checkpoint_interval}"
        storage_info = f"Store directly to ChromaDB collection: {collection_name}"

        print_stage_header(
            4,
            input_file=f"documents_chunked.parquet ({input_count:,} chunks)",
            output_file=str(chroma_path),
            action=f"{action}\n{batch_info}\n{storage_info}",
            current_stage=current_stage,
            total_stages=total_stages,
        )

        try:
            console.print(f"[cyan]🔮 Starting embedding generation for {input_count:,} chunks...[/cyan]")
            console.print(f"[dim]Model: {self.config.embedding_model} ({self.config.embedding_dimensions}D)[/dim]")
            console.print(f"[dim]Target collection: {collection_name}[/dim]")

            # Set up ChromaDB collection
            console.print("[cyan]🗃️ Setting up ChromaDB collection...[/cyan]")
            collection_metadata = {
                "dataset": self.config.dataset_name,
                "chunk_size": self.config.chunk_size,
                "embedding_model": self.config.embedding_model,
                "embedding_dimensions": self.config.embedding_dimensions,
                "created_at": datetime.now().isoformat(),
                "total_documents": input_count,
            }

            collection = self.chroma_manager.get_or_create_collection(collection_name, metadata=collection_metadata)
            console.print(f"[green]✅ Collection '{collection_name}' ready[/green]")

            # Calculate batch information
            total_batches = (input_count + self.config.embedding_batch_size - 1) // self.config.embedding_batch_size

            # Prepare data for processing
            document_ids = input_data.select("unique_id").to_series().to_list()
            texts = input_data.select("chunked_prompt").to_series().to_list()

            # Process and store in batches
            processed_count = 0
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Generating and storing embeddings..."),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("| {task.completed}/{task.total} batches"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                embedding_task = progress.add_task("Processing", total=total_batches)
                batch_start_time = time.time()

                for batch_idx in range(total_batches):
                    batch_start = batch_idx * self.config.embedding_batch_size
                    batch_end = min(batch_start + self.config.embedding_batch_size, input_count)
                    batch_texts = texts[batch_start:batch_end]
                    batch_ids = document_ids[batch_start:batch_end]

                    # Generate embeddings for this batch
                    batch_embeddings = self.embedding_generator.generate_embeddings_batch(
                        batch_texts, self.config.embedding_model, dimensions=self.config.embedding_dimensions
                    )

                    # Store directly to ChromaDB
                    collection.add(ids=batch_ids, documents=batch_texts, embeddings=batch_embeddings)

                    processed_count += len(batch_texts)

                    # Update progress
                    elapsed = time.time() - batch_start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    progress.update(embedding_task, advance=1, description=f"Processing | {rate:.1f} docs/sec")

            console.print(f"[green]✅ Stored {processed_count:,} documents with embeddings to ChromaDB[/green]")

            # Calculate statistics
            stage_time = time.time() - stage_start_time

            stats = {
                "chunks_processed": input_count,
                "documents_stored": processed_count,
                "collection_name": collection_name,
                "embedding_dimension": self.config.embedding_dimensions,
                "total_batches": total_batches,
                "documents_per_second": round(processed_count / stage_time, 1),
                "model_used": self.config.embedding_model,
                "elapsed_time": stage_time,
            }

            # Print completion
            print_stage_complete(4, stats, saved_to=f"ChromaDB collection: {collection_name}", next_stage=5)

            # Save checkpoint with ChromaDB collection info
            checkpoint_file = self.config.processed_dir / "stage4_chromadb_complete.json"
            self.checkpoint.save_checkpoint(4, output_file=checkpoint_file, stats=stats)

            logger.info(f"Stage 4 completed: {processed_count:,} documents stored to ChromaDB in {stage_time:.1f}s")

            return input_data, {"status": "completed", **stats}

        except Exception as e:
            logger.error(f"Stage 4 failed: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def _run_stage_5(
        self, input_data: pl.DataFrame, current_stage: int = 1, total_stages: int = 6
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Stage 5: LLM Filtering from ChromaDB"""
        stage_start_time = time.time()

        # Get ChromaDB collection info
        collection_name = f"wikitext_{self.config.dataset_name.replace('/', '_').replace('-', '_')}"

        # Handle resumption from checkpoint - get collection if needed
        if input_data is None:
            console.print(f"[cyan]📖 Loading documents from ChromaDB collection: {collection_name}[/cyan]")

            # Get the collection
            try:
                collection = self.chroma_manager.client.get_collection(collection_name)

                # Get all documents from the collection for filtering
                all_docs = collection.get(include=["documents", "metadatas"])

                # Convert to DataFrame for processing
                if all_docs["ids"]:
                    input_data = pl.DataFrame(
                        {
                            "unique_id": all_docs["ids"],
                            "chunked_prompt": all_docs["documents"],
                        }
                    )
                    logger.info(f"Loaded {len(input_data):,} documents from ChromaDB collection")
                else:
                    logger.error(f"Stage 5 cannot resume: No documents found in ChromaDB collection '{collection_name}'")
                    raise RuntimeError(f"No documents found in ChromaDB collection '{collection_name}'")

            except Exception as e:
                logger.error(f"Stage 5 cannot resume: Failed to load from ChromaDB collection '{collection_name}': {e}")
                raise RuntimeError(
                    f"Stage 5 cannot resume: Failed to load from ChromaDB collection '{collection_name}': {e}"
                )

        input_count = len(input_data)
        output_file = self.config.processed_dir / "filtered_documents.parquet"

        # Build action description with filtering details
        action_parts = [f"Filter chunks using {self.config.filter_model}"]
        action_parts.append(f"Source: ChromaDB collection '{collection_name}'")
        action_parts.append(f"Approach: Consecutive chunks with question generation")

        if self.config.filter_samples_per_doc or self.config.filter_max_total_chunks:
            limits = []
            if self.config.filter_samples_per_doc:
                limits.append(f"{self.config.filter_samples_per_doc} per doc")
            if self.config.filter_max_total_chunks:
                limits.append(f"{self.config.filter_max_total_chunks} total")
            action_parts.append(f"Sampling: {', '.join(limits)}")

        action = "\n".join(action_parts)

        print_stage_header(
            5,
            input_file=f"ChromaDB: {collection_name} ({input_count:,} chunks)",
            output_file=str(output_file),
            action=action,
            current_stage=current_stage,
            total_stages=total_stages,
        )

        try:
            # Apply filtering limits if specified
            console.print("[cyan]🔍 Preparing chunks for LLM filtering...[/cyan]")

            # Apply the filtering limits we implemented earlier
            sampled_data = self._apply_filtering_limits(input_data)
            sampled_count = len(sampled_data)

            if sampled_count != input_count:
                console.print(
                    f"[yellow]📊 Applied sampling limits: {input_count:,} → {sampled_count:,} chunks[/yellow]"
                )

            # Run LLM filtering with async processing
            console.print(f"[cyan]Starting LLM filtering with {self.config.filter_model}...[/cyan]")
            console.print(f"[dim]Concurrency: {self.config.max_concurrent_requests} requests[/dim]")

            # Run the async filtering pipeline - this is the only async part
            async def run_async_filtering():
                checkpoint_path = Path("logs") / "filtering_checkpoint.json"

                filtered_df, filter_metadata = await self.document_filter.run_filtering_pipeline(
                    sampled_data,
                    use_async=True,
                    batch_size=self.config.filter_batch_size,
                    checkpoint_path=str(checkpoint_path),
                )
                return filtered_df, filter_metadata

            # Run the async function
            filtered_df, filter_metadata = asyncio.run(run_async_filtering())

            console.print("[green]✅ Filtering completed[/green]")

            # Save filtered documents to file
            with Progress(
                SpinnerColumn(), TextColumn("[bold blue]Saving filtered documents..."), console=console
            ) as progress:
                save_task = progress.add_task("Saving", total=100)

                # Ensure output directory exists
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # Save filtered documents
                filtered_df.write_parquet(output_file)
                progress.update(save_task, advance=100)

            # Calculate statistics
            final_count = len(filtered_df)
            stage_time = time.time() - stage_start_time

            stats = {
                "chunks_input": input_count,
                "chunks_sampled": sampled_count,
                "chunks_filtered": final_count,
                "filtering_applied": sampled_count != input_count,
                "pass_rate": f"{100 * final_count / sampled_count:.1f}%",
                "overall_pass_rate": f"{100 * final_count / input_count:.1f}%",
                "filter_model": self.config.filter_model,
                "consecutive_chunks_count": self.config.consecutive_chunks_count,
                "batch_size": self.config.filter_batch_size,
                "elapsed_time": stage_time,
                **filter_metadata,  # Include detailed filtering metadata
            }

            # Print completion
            print_stage_complete(5, stats, saved_to=str(output_file), next_stage=6)

            # Save checkpoint
            self.checkpoint.save_checkpoint(5, output_file=output_file, stats=stats)

            logger.info(
                f"Stage 5 completed: {final_count:,} chunks passed filtering ({sampled_count:,} evaluated) in {stage_time:.1f}s"
            )

            return filtered_df, {"status": "completed", **stats}

        except Exception as e:
            logger.error(f"Stage 5 failed: {e}")
            raise RuntimeError(f"LLM filtering failed: {e}") from e

    def _apply_filtering_limits(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply filtering limits to control batch sizes for LLM processing."""
        original_count = len(df)

        # Apply per-document sampling first if specified
        if self.config.filter_samples_per_doc is not None:
            logger.info(f"Applying per-document sampling: max {self.config.filter_samples_per_doc} chunks per document")

            # Group by document and sample chunks within each document
            if "document_url" in df.columns:
                # Safe sampling: use min(requested_samples, available_chunks) per document
                df = (
                    df.group_by("document_url")
                    .agg(
                        [
                            pl.all().sample(
                                n=pl.min_horizontal(
                                    pl.lit(self.config.filter_samples_per_doc), pl.col("unique_id").len()
                                )
                            )
                        ]
                    )
                    .explode(pl.exclude("document_url"))
                )
                logger.info(f"After per-document sampling: {len(df)} chunks ({original_count - len(df)} removed)")

        # Apply total chunk limit if specified
        if self.config.filter_max_total_chunks is not None:
            if len(df) > self.config.filter_max_total_chunks:
                logger.info(
                    f"Applying total chunk limit: sampling {self.config.filter_max_total_chunks} from {len(df)} chunks"
                )
                # Use min() to prevent sampling more than available
                sample_size = min(self.config.filter_max_total_chunks, len(df))
                df = df.sample(sample_size)
                logger.info(f"After total sampling: {len(df)} chunks")

        if len(df) != original_count:
            logger.info(
                f"📊 Filtering limits applied: {original_count} → {len(df)} chunks ({100 * len(df) / original_count:.1f}%)"
            )

        return df

    def _run_stage_6(
        self, input_data: pl.DataFrame, current_stage: int = 1, total_stages: int = 6
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Stage 6: ChromaDB Metadata Update"""
        stage_start_time = time.time()

        # Handle resumption from checkpoint - load filtered documents if needed
        if input_data is None:
            filtered_file = self.config.processed_dir / "filtered_documents.parquet"
            if not filtered_file.exists():
                logger.error(f"Stage 6 cannot resume: filtered_documents.parquet not found at {filtered_file}")
                raise RuntimeError("Stage 6 cannot resume: filtered_documents.parquet not found")
            input_data = pl.read_parquet(filtered_file)
            logger.info(f"Loaded {len(input_data):,} filtered chunks from {filtered_file}")

        input_count = len(input_data)

        # Generate collection name
        collection_name = f"wikitext_{self.config.dataset_name.replace('/', '_').replace('-', '_')}"
        chroma_path = self.config.chroma_db_path / collection_name

        # Calculate passed/failed chunks
        passed_ids = input_data.select("unique_id").to_series().to_list()

        action = f"Update ChromaDB collection with quality metadata\nCollection: {collection_name}"
        metadata_info = f"Mark {len(passed_ids):,} chunks as quality-filtered (passed)"

        print_stage_header(
            6,
            input_file=f"filtered_documents.parquet ({input_count:,} chunks)",
            output_file=str(chroma_path),
            action=f"{action}\n{metadata_info}",
            current_stage=current_stage,
            total_stages=total_stages,
        )

        try:
            console.print("[cyan]🗃️ Starting ChromaDB metadata update...[/cyan]")
            console.print(f"[dim]Collection: {collection_name}[/dim]")
            console.print(f"[dim]Database path: {self.config.chroma_db_path}[/dim]")

            # Get existing ChromaDB collection
            with Progress(
                SpinnerColumn(), TextColumn("[bold blue]Accessing ChromaDB collection..."), console=console
            ) as progress:
                setup_task = progress.add_task("Setup", total=100)

                try:
                    collection = self.chroma_manager.client.get_collection(collection_name)
                    progress.update(setup_task, advance=100)
                except Exception as e:
                    logger.error(f"Stage 6 cannot access ChromaDB collection '{collection_name}': {e}")
                    logger.error("Ensure Stage 4 (Embedding Generation) completed successfully")
                    raise RuntimeError(
                        f"Cannot access ChromaDB collection '{collection_name}': {e}. Ensure Stage 4 completed."
                    )

            console.print(f"[green]✅ Collection '{collection_name}' accessed[/green]")

            # Get total count before filtering
            total_collection_count = collection.count()
            console.print(f"[cyan]📊 Total documents in collection: {total_collection_count:,}[/cyan]")
            console.print(
                f"[cyan]📊 Documents that passed filtering: {input_count:,} ({100 * input_count / total_collection_count:.1f}%)[/cyan]"
            )

            # Update metadata for passed documents
            console.print(f"[cyan]Updating metadata for {input_count:,} quality-filtered documents...[/cyan]")

            # Add metadata indicating these documents passed quality filtering
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Updating quality metadata..."),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                update_task = progress.add_task("Updating", total=len(passed_ids))

                batch_size = 500
                updated_count = 0

                for i in range(0, len(passed_ids), batch_size):
                    batch_ids = passed_ids[i : i + batch_size]

                    # Update metadata for this batch
                    batch_metadata = [
                        {
                            "quality_filtered": True,
                            "filter_model": self.config.filter_model,
                            "filtered_at": datetime.now().isoformat(),
                        }
                    ] * len(batch_ids)

                    collection.update(ids=batch_ids, metadatas=batch_metadata)

                    updated_count += len(batch_ids)
                    progress.update(update_task, advance=len(batch_ids))

            console.print(f"[green]✅ Metadata updated for {updated_count:,} documents[/green]")

            # Final verification
            final_count = input_count

            # Calculate statistics
            stage_time = time.time() - stage_start_time

            stats = {
                "documents_updated": final_count,
                "total_collection_size": total_collection_count,
                "quality_pass_rate": round(100 * final_count / total_collection_count, 1),
                "collection_name": collection_name,
                "collection_path": str(chroma_path),
                "database_size": str(self.config.chroma_db_path),
                "documents_per_second": round(final_count / stage_time, 1),
                "elapsed_time": stage_time,
            }

            # Create completion message with query instructions
            completion_stats = {
                **stats,
                "query_instructions": f"Use: chromadb.PersistentClient('{self.config.chroma_db_path}').get_collection('{collection_name}')",
            }

            # Print completion
            print_stage_complete(6, completion_stats, saved_to=f"ChromaDB collection: {collection_name}")

            # Save checkpoint
            checkpoint_file = self.config.processed_dir / "stage6_metadata_complete.json"
            self.checkpoint.save_checkpoint(6, output_file=checkpoint_file, stats=stats)

            # Print final usage instructions
            console.print()
            console.print(
                Panel(
                    f"🎉 ChromaDB Collection Updated!\n\n"
                    f"Collection: [bold cyan]{collection_name}[/bold cyan]\n"
                    f"Total Documents: [yellow]{total_collection_count:,}[/yellow]\n"
                    f"Quality-Filtered: [green]{final_count:,}[/green] ([cyan]{100 * final_count / total_collection_count:.1f}%[/cyan])\n"
                    f"Path: [green]{self.config.chroma_db_path}[/green]\n\n"
                    f"[bold]Query Example:[/bold]\n"
                    f"[dim]```python[/dim]\n"
                    f"[dim]import chromadb[/dim]\n"
                    f"[dim]client = chromadb.PersistentClient('{self.config.chroma_db_path}')[/dim]\n"
                    f"[dim]collection = client.get_collection('{collection_name}')[/dim]\n"
                    f"[dim]# Query only quality-filtered documents:[/dim]\n"
                    f"[dim]results = collection.query([/dim]\n"
                    f"[dim]    query_texts=['your query'],[/dim]\n"
                    f"[dim]    n_results=5,[/dim]\n"
                    f'[dim]    where={{"quality_filtered": True}}[/dim]\n'
                    f"[dim])[/dim]\n"
                    f"[dim]```[/dim]",
                    title="✅ Metadata Update Complete",
                    style="green",
                    padding=(1, 2),
                    width=80,
                )
            )

            logger.info(
                f"Stage 6 completed: {final_count:,} documents updated with quality metadata in '{collection_name}' in {stage_time:.1f}s"
            )

            return input_data, {"status": "completed", **stats}

        except Exception as e:
            logger.error(f"Stage 6 failed: {e}")
            raise RuntimeError(f"ChromaDB ingestion failed: {e}") from e
