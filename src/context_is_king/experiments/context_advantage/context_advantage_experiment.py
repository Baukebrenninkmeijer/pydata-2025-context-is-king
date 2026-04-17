#!/usr/bin/env python3
"""
Context Window Advantage Experiment Coordinator

🎯 DATASETS: Synthetic Needles + LongMemEval Conversations
📊 PURPOSE: Do larger context windows outperform smaller ones at same context size?

Main experiment controller that coordinates both needle-in-haystack and LongMemEval
experiments to test whether models with larger context windows outperform models
with smaller context windows at the same context window size.

DATASET USAGE:
- Needle-in-haystack: Synthetic needles + Paul Graham essays + ArXiv papers
- LongMemEval validation: Real conversation data for ground truth validation
- Fixed Q&A sets: Pre-generated questions for consistent evaluation

Usage:
    python context_advantage_experiment.py --experiment-type needle --context-sizes 10000,50000,100000
    python context_advantage_experiment.py --experiment-type longmemeval --models claude-sonnet-4,claude-haiku-3.5
    python context_advantage_experiment.py --full-experiment
"""

import argparse
import json
import os
import signal
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dotenv
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# Import from relative modules
from .longmemeval_analysis.context_grouper import LongMemEvalContextGrouper
from .needle_haystack.haystack_builder import Haystack, HaystackBuilder
from .needle_haystack.needle_generator import NeedleGenerator, NeedleSet

from context_is_king.evaluation import LongMemEvalEvaluator, NeedleEvaluator
from context_is_king.models import ModelInterface, QueryResult

dotenv.load_dotenv(dotenv_path=Path(__file__).parents[2] / ".env", override=True)


@dataclass
class NeedlePosition:
    """Tracks the exact position of a needle in the haystack."""

    needle_id: str
    section_index: int
    section_ratio: float
    position_zone: str  # 'beginning', 'early_middle', 'center', 'late_middle', 'end'
    char_start: int  # Character position where needle starts
    char_end: int  # Character position where needle ends
    token_start: int  # Approximate token position where needle starts
    token_end: int  # Approximate token position where needle ends


@dataclass
class QuestionNeedleMapping:
    """Maps questions to their associated needles and positions."""

    question_id: str
    needle_ids: list[str]  # Needles required to answer this question
    primary_position: NeedlePosition  # Position of primary needle
    secondary_positions: list[NeedlePosition]  # Positions of additional needles if any


@dataclass
class ExperimentProgress:
    """Tracks overall experiment progress with detailed statistics."""

    # Experiment overview
    total_needle_experiments: int
    total_longmem_experiments: int
    total_experiments: int
    total_llm_calls: int
    estimated_total_cost: float
    estimated_duration_minutes: float

    # Current progress
    completed_needle: int = 0
    completed_longmem: int = 0
    completed_total: int = 0
    completed_llm_calls: int = 0
    actual_cost: float = 0.0

    # Timing
    start_time: float = 0.0
    current_phase: str = "needle"
    phase_start_time: float = 0.0

    def __post_init__(self):
        self.start_time = time.time()
        self.phase_start_time = time.time()

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def phase_elapsed_time(self) -> float:
        """Elapsed time for current phase in seconds."""
        return time.time() - self.phase_start_time

    @property
    def completion_percentage(self) -> float:
        """Overall completion percentage."""
        return (self.completed_total / self.total_experiments * 100) if self.total_experiments > 0 else 0

    @property
    def estimated_remaining_minutes(self) -> float:
        """Estimated remaining time in minutes."""
        if self.completed_total == 0:
            return self.estimated_duration_minutes

        elapsed_minutes = self.elapsed_time / 60
        rate = self.completed_total / elapsed_minutes if elapsed_minutes > 0 else 0
        remaining = self.total_experiments - self.completed_total

        return (remaining / rate) if rate > 0 else self.estimated_duration_minutes


class ProgressTracker:
    """Manages rich-enhanced progress bars and experiment progress display."""

    def __init__(self, progress: ExperimentProgress):
        self.progress = progress
        self.console = Console()
        self.rich_progress = None
        self.main_task = None
        self.experiment_task = None
        self.current_experiment_name = ""

    def start_experiment(self, config):
        """Initialize rich-enhanced progress display and experiment overview."""
        self.console.print()

        # Create title panel
        title = Panel(
            Align.center(Text("🚀 CONTEXT ADVANTAGE EXPERIMENT STARTING", style="bold cyan")),
            style="cyan",
            box=box.DOUBLE,
        )
        self.console.print(title)

        # Create experiment overview table
        overview_table = Table(title="📋 Experiment Overview", box=box.ROUNDED, title_style="bold blue")
        overview_table.add_column("Property", style="cyan", no_wrap=True)
        overview_table.add_column("Value", style="white")

        overview_table.add_row("Experiment ID", f"[bold yellow]{config.experiment_id}[/bold yellow]")
        overview_table.add_row("Type", f"[bold green]{config.experiment_type}[/bold green]")
        overview_table.add_row("Models", f"[bold magenta]{', '.join(config.models)}[/bold magenta]")

        if "needle" in config.experiment_type or config.experiment_type == "both":
            overview_table.add_row("Context Sizes", f"[cyan]{', '.join(map(str, config.context_sizes))}[/cyan]")
            overview_table.add_row("Compositions", f"[yellow]{', '.join(config.needle_compositions)}[/yellow]")
            overview_table.add_row("Iterations", f"[green]{config.iterations}[/green]")
            overview_table.add_row("Needles per haystack", f"[red]{config.needles_per_haystack}[/red]")

        if "longmemeval" in config.experiment_type or config.experiment_type == "both":
            overview_table.add_row("Question Groups", f"[blue]{', '.join(config.question_groups)}[/blue]")

        self.console.print(overview_table)

        # Add detailed needle positioning information with rich formatting
        if "needle" in config.experiment_type or config.experiment_type == "both":
            self._display_needle_positioning_info_rich(config)

        # Create statistics table
        stats_table = Table(title="📊 Experiment Statistics", box=box.ROUNDED, title_style="bold green")
        stats_table.add_column("Metric", style="cyan", no_wrap=True)
        stats_table.add_column("Value", style="white")

        stats_table.add_row("Total Experiments", f"[bold yellow]{self.progress.total_experiments:,}[/bold yellow]")
        stats_table.add_row("• Needle Experiments", f"[green]{self.progress.total_needle_experiments:,}[/green]")
        stats_table.add_row("• LongMemEval Experiments", f"[blue]{self.progress.total_longmem_experiments:,}[/blue]")
        stats_table.add_row("Total LLM Calls", f"[bold red]{self.progress.total_llm_calls:,}[/bold red]")
        stats_table.add_row("Estimated Cost", f"[bold green]${self.progress.estimated_total_cost:.2f}[/bold green]")
        stats_table.add_row(
            "Estimated Duration", f"[cyan]{self.progress.estimated_duration_minutes:.1f} minutes[/cyan]"
        )

        self.console.print(stats_table)

        # Create timing panel
        start_time = time.strftime("%H:%M:%S", time.localtime(self.progress.start_time))
        end_time = time.strftime(
            "%H:%M:%S", time.localtime(self.progress.start_time + self.progress.estimated_duration_minutes * 60)
        )

        timing_content = f"⏱️  [bold]Start Time:[/bold] [green]{start_time}[/green]\n⏰ [bold]Expected End:[/bold] [yellow]{end_time}[/yellow]"
        timing_panel = Panel(timing_content, title="⏱️ Timing", box=box.ROUNDED, style="blue")
        self.console.print(timing_panel)

        # Initialize fancy rich progress bars
        self.rich_progress = Progress(
            SpinnerColumn(),
            TextColumn("🧪 [bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
        )

        # Start the progress display
        self.rich_progress.start()

        # Add main overall progress task
        self.main_task = self.rich_progress.add_task("Overall Progress", total=self.progress.total_experiments)

    def start_phase(self, phase: str, total_experiments: int):
        """Start a new experiment phase."""
        self.progress.current_phase = phase
        self.progress.phase_start_time = time.time()

        # Update main progress description with current phase
        phase_icon = "🎯" if phase == "needle" else "📚"
        phase_name = "Needle-in-Haystack" if phase == "needle" else "LongMemEval"

        if self.main_task is not None:
            self.rich_progress.update(self.main_task, description=f"Overall Progress • {phase_icon} {phase_name}")

    def start_experiment_task(self, experiment_description: str):
        """Start a new individual experiment task with detailed progress."""
        self.current_experiment_name = experiment_description

        # Remove previous experiment task if exists
        if self.experiment_task is not None:
            self.rich_progress.remove_task(self.experiment_task)

        # Add new experiment task
        self.experiment_task = self.rich_progress.add_task(f"🔬 {experiment_description}", total=1)

    def update_experiment_task(self, status: str, progress: float = None):
        """Update the current experiment task with status."""
        if self.experiment_task is not None:
            description = f"🔬 {status}"
            if progress is not None:
                self.rich_progress.update(self.experiment_task, completed=progress, description=description)
            else:
                self.rich_progress.update(self.experiment_task, description=description)

    def complete_experiment_task(self):
        """Mark the current experiment as completed."""
        if self.experiment_task is not None:
            self.rich_progress.update(self.experiment_task, completed=1)
            self.rich_progress.remove_task(self.experiment_task)
            self.experiment_task = None

    def update_experiment_progress(self, result, phase: str):
        """Update progress after completing an experiment."""
        # Update counters
        if phase == "needle":
            self.progress.completed_needle += 1
        else:
            self.progress.completed_longmem += 1

        self.progress.completed_total += 1

        # Estimate LLM calls and cost based on result
        if hasattr(result, "total_cost"):
            self.progress.actual_cost += result.total_cost

        # Update main progress bar
        if self.main_task is not None:
            completion_pct = self.progress.completion_percentage
            remaining_min = self.progress.estimated_remaining_minutes

            # Get current phase info
            phase_icon = "🎯" if phase == "needle" else "📚"
            phase_name = "Needle" if phase == "needle" else "LongMem"

            description = (
                f"Overall Progress ({completion_pct:.1f}%) • "
                f"{phase_icon} {phase_name} • "
                f"${self.progress.actual_cost:.2f}"
            )

            self.rich_progress.update(self.main_task, advance=1, description=description)

    def display_phase_summary(self, phase: str, results):
        """Display rich-formatted summary after completing a phase."""
        # No need to close phase_pbar since we're not using it anymore

        phase_name = "Needle-in-Haystack" if phase == "needle" else "LongMemEval"
        phase_icon = "🎯" if phase == "needle" else "📚"
        phase_color = "red" if phase == "needle" else "blue"

        if results:
            avg_score = sum(r.success_rate for r in results) / len(results)
            avg_time = sum(r.avg_response_time for r in results) / len(results)
            total_cost = sum(r.total_cost for r in results)

            # Create phase completion table
            summary_table = Table(
                title=f"{phase_icon} {phase_name} Phase Completed", box=box.ROUNDED, title_style=f"bold {phase_color}"
            )
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="white")

            summary_table.add_row("✅ Experiments", f"[bold yellow]{len(results)}[/bold yellow]")
            summary_table.add_row("📊 Avg Success Rate", f"[bold green]{avg_score:.1%}[/bold green]")
            summary_table.add_row("⏱️ Avg Response Time", f"[cyan]{avg_time:.2f}s[/cyan]")
            summary_table.add_row("💰 Phase Cost", f"[bold green]${total_cost:.2f}[/bold green]")

            self.console.print(summary_table)

    def finish_experiment(self):
        """Clean up progress bars and display rich final summary."""
        # Stop the rich progress display
        if self.rich_progress:
            self.rich_progress.stop()

        # Create completion title panel
        completion_title = Panel(
            Align.center(Text("✅ EXPERIMENT COMPLETED", style="bold green")), style="green", box=box.DOUBLE
        )
        self.console.print(completion_title)

        elapsed_minutes = self.progress.elapsed_time / 60

        # Create final statistics table
        final_table = Table(title="📊 Final Statistics", box=box.ROUNDED, title_style="bold green")
        final_table.add_column("Metric", style="cyan")
        final_table.add_column("Value", style="white")

        final_table.add_row(
            "Total Experiments",
            f"[bold yellow]{self.progress.completed_total:,}/{self.progress.total_experiments:,}[/bold yellow]",
        )
        final_table.add_row("• Needle Experiments", f"[red]{self.progress.completed_needle:,}[/red]")
        final_table.add_row("• LongMemEval Experiments", f"[blue]{self.progress.completed_longmem:,}[/blue]")
        final_table.add_row("Total Duration", f"[cyan]{elapsed_minutes:.1f} minutes[/cyan]")
        final_table.add_row("Total Cost", f"[bold green]${self.progress.actual_cost:.2f}[/bold green]")

        if elapsed_minutes > 0:
            rate = self.progress.completed_total / elapsed_minutes
            final_table.add_row("Average Rate", f"[yellow]{rate:.2f} experiments/minute[/yellow]")

        self.console.print(final_table)

        # Create success celebration panel
        success_content = """[bold green]🎉[/bold green] All experiments completed successfully!
[bold blue]📈[/bold blue] Results ready for analysis
[bold yellow]💾[/bold yellow] Data saved to output directory"""

        success_panel = Panel(success_content, title="🎉 Success!", box=box.ROUNDED, style="green")
        self.console.print(success_panel)

    def _display_needle_positioning_info_rich(self, config):
        """Display detailed needle positioning information using rich formatting."""

        # Needle positioning strategy table
        positioning_table = Table(title="📍 Needle Positioning Strategy", box=box.ROUNDED, title_style="bold red")
        positioning_table.add_column("Zone", style="bold cyan", no_wrap=True)
        positioning_table.add_column("Range", style="yellow")
        positioning_table.add_column("Needles", style="green", justify="center")
        positioning_table.add_column("Description", style="white")

        # Define position zones and their characteristics
        position_zones = [
            ("Beginning", "0-10%", "Start of context"),
            ("Early Middle", "10-35%", "Early section"),
            ("Center", "35-65%", "Middle section"),
            ("Late Middle", "65-85%", "Late section"),
            ("End", "85-100%", "End of context"),
        ]

        needles_per_zone = config.needles_per_haystack // len(position_zones)
        remaining_needles = config.needles_per_haystack % len(position_zones)

        # Calculate needle distribution
        zone_needle_counts = [needles_per_zone] * len(position_zones)
        for i in range(remaining_needles):
            zone_needle_counts[i] += 1

        for (zone, ratio, description), count in zip(position_zones, zone_needle_counts, strict=False):
            if count > 0:
                needle_text = f"{count} needle{'s' if count > 1 else ''}"
                positioning_table.add_row(zone, ratio, needle_text, description)

        self.console.print(positioning_table)

        # Position tracking capabilities
        tracking_content = """[bold green]✅[/bold green] Exact character positions (start/end)
[bold green]✅[/bold green] Estimated token positions (start/end)
[bold green]✅[/bold green] Section index and ratio within haystack
[bold green]✅[/bold green] Position zone classification
[bold green]✅[/bold green] Question-to-needle mappings"""

        tracking_panel = Panel(
            tracking_content, title="📊 Position Tracking Per Needle", box=box.ROUNDED, style="green"
        )
        self.console.print(tracking_panel)

        # Analysis capabilities
        analysis_content = """[bold blue]📈[/bold blue] Performance by zone (success rates, scores)
[bold blue]📍[/bold blue] Needle distribution statistics
[bold blue]🎯[/bold blue] Position accuracy validation
[bold blue]⚖️[/bold blue] Zone bias detection"""

        analysis_panel = Panel(analysis_content, title="🔬 Position-Based Analysis", box=box.ROUNDED, style="blue")
        self.console.print(analysis_panel)

        # Context size variations table
        if len(config.context_sizes) > 1:
            size_table = Table(title="📏 Context Size Variations", box=box.ROUNDED, title_style="bold magenta")
            size_table.add_column("Context Size", style="cyan")
            size_table.add_column("Tokens per Zone", style="yellow")

            for size in config.context_sizes:
                approx_tokens_per_zone = size // len(position_zones)
                size_table.add_row(f"{size:,} tokens", f"~{approx_tokens_per_zone:,} tokens")

            self.console.print(size_table)

        # Example positioning for largest context
        if config.context_sizes:
            max_context = max(config.context_sizes)

            example_table = Table(
                title=f"📍 Example Positioning ({max_context:,} token context)",
                box=box.ROUNDED,
                title_style="bold yellow",
            )
            example_table.add_column("Zone", style="bold cyan")
            example_table.add_column("Token Range", style="yellow")
            example_table.add_column("Needles", style="green")

            for (zone, ratio, _), count in zip(position_zones, zone_needle_counts, strict=False):
                if count > 0:
                    # Calculate approximate token ranges
                    start_pct = float(ratio.split("-")[0].rstrip("%")) / 100
                    end_pct = float(ratio.split("-")[1].rstrip("%")) / 100
                    start_token = int(max_context * start_pct)
                    end_token = int(max_context * end_pct)

                    needle_text = f"{count} needle{'s' if count > 1 else ''}"
                    example_table.add_row(zone, f"~{start_token:,}-{end_token:,}", needle_text)

            self.console.print(example_table)

        # Question types and requirements
        question_table = Table(
            title="❓ Question Types & Needle Requirements", box=box.ROUNDED, title_style="bold purple"
        )
        question_table.add_column("Type", style="bold cyan")
        question_table.add_column("Needles", style="red", justify="center")
        question_table.add_column("Description", style="white")

        question_info = [
            ("Direct", "1", "Simple fact retrieval from single location"),
            ("Cross Reference", "2", "Connect information from multiple positions"),
            ("Synthesis", "3", "Combine facts from across the context"),
            ("Domain Transfer", "2", "Apply concepts between different domains"),
        ]

        for qtype, needle_req, description in question_info:
            question_table.add_row(qtype, needle_req, description)

        self.console.print(question_table)

        # Question distribution info
        questions_per_needle = 2
        total_questions = config.needles_per_haystack * questions_per_needle

        distribution_content = f"""[bold]•[/bold] [cyan]{questions_per_needle}[/cyan] questions per needle
[bold]•[/bold] [yellow]{total_questions}[/yellow] total questions per experiment
[bold]•[/bold] Questions use [green]fixed Q&As[/green] (40 predefined per type)
[bold]•[/bold] Needles distributed [blue]strategically[/blue] across positions"""

        distribution_panel = Panel(
            distribution_content, title="🔢 Question Distribution", box=box.ROUNDED, style="cyan"
        )
        self.console.print(distribution_panel)

        # Benefits
        benefits_content = """[bold green]🎯[/bold green] Tests model attention across full context window
[bold green]📊[/bold green] Enables position bias analysis
[bold green]🔍[/bold green] Validates consistent retrieval performance
[bold green]⚡[/bold green] Identifies optimal placement strategies
[bold green]🧠[/bold green] Reveals attention patterns and context utilization
[bold green]📈[/bold green] Quantifies performance degradation by position"""

        benefits_panel = Panel(benefits_content, title="💡 Positioning Benefits", box=box.ROUNDED, style="green")
        self.console.print(benefits_panel)

    def _display_needle_positioning_info(self, config):
        """Display detailed information about needle positioning strategy."""
        print("\n📍 NEEDLE POSITIONING STRATEGY:")

        # Define position zones and their characteristics
        position_zones = [
            ("beginning", "0-10%", "Start of context"),
            ("early_middle", "10-35%", "Early section"),
            ("center", "35-65%", "Middle section"),
            ("late_middle", "65-85%", "Late section"),
            ("end", "85-100%", "End of context"),
        ]

        needles_per_zone = config.needles_per_haystack // len(position_zones)
        remaining_needles = config.needles_per_haystack % len(position_zones)

        # Calculate needle distribution
        zone_needle_counts = [needles_per_zone] * len(position_zones)
        for i in range(remaining_needles):
            zone_needle_counts[i] += 1

        print(f"   Distribution Strategy: {config.needles_per_haystack} needles across 5 zones")

        for i, ((zone, ratio, description), count) in enumerate(zip(position_zones, zone_needle_counts, strict=False)):
            if count > 0:
                print(
                    f"   • {zone.replace('_', ' ').title()}: {count} needle{'s' if count > 1 else ''} ({ratio} of context) - {description}"
                )

        # Show what gets tracked for each needle
        print("\n📊 POSITION TRACKING PER NEEDLE:")
        print("   ✅ Exact character positions (start/end)")
        print("   ✅ Estimated token positions (start/end)")
        print("   ✅ Section index and ratio within haystack")
        print("   ✅ Position zone classification")
        print("   ✅ Question-to-needle mappings")

        # Show analysis capabilities
        print("\n🔬 POSITION-BASED ANALYSIS:")
        print("   📈 Performance by zone (success rates, scores)")
        print("   📍 Needle distribution statistics")
        print("   🎯 Position accuracy validation")
        print("   ⚖️ Zone bias detection")

        # Context size specific information
        if len(config.context_sizes) > 1:
            print("\n📏 CONTEXT SIZE VARIATIONS:")
            for size in config.context_sizes:
                approx_tokens_per_zone = size // len(position_zones)
                print(f"   • {size:,} tokens: ~{approx_tokens_per_zone:,} tokens per zone")

        # Show example positioning for largest context
        if config.context_sizes:
            max_context = max(config.context_sizes)
            print(f"\n📍 EXAMPLE POSITIONING ({max_context:,} token context):")

            for i, ((zone, ratio, description), count) in enumerate(
                zip(position_zones, zone_needle_counts, strict=False)
            ):
                if count > 0:
                    # Calculate approximate token ranges
                    start_pct = float(ratio.split("-")[0].rstrip("%")) / 100
                    end_pct = float(ratio.split("-")[1].rstrip("%")) / 100
                    start_token = int(max_context * start_pct)
                    end_token = int(max_context * end_pct)

                    print(
                        f"   • {zone.replace('_', ' ').title()}: tokens ~{start_token:,}-{end_token:,} ({count} needle{'s' if count > 1 else ''})"
                    )

        # Add question type information
        print("\n❓ QUESTION TYPES & NEEDLE REQUIREMENTS:")
        question_info = [
            ("Direct", "1 needle", "Simple fact retrieval from single location"),
            ("Cross Reference", "2 needles", "Connect information from multiple positions"),
            ("Synthesis", "3 needles", "Combine facts from across the context"),
            ("Domain Transfer", "2 needles", "Apply concepts between different domains"),
        ]

        for qtype, needle_req, description in question_info:
            print(f"   • {qtype}: {needle_req} - {description}")

        questions_per_needle = 2  # Default from needle generator
        total_questions = config.needles_per_haystack * questions_per_needle
        print("\n🔢 QUESTION DISTRIBUTION:")
        print(f"   • {questions_per_needle} questions per needle")
        print(f"   • {total_questions} total questions per experiment")
        print("   • Questions use fixed Q&As (40 predefined per type)")
        print("   • Needles distributed strategically across positions")

        print("\n💡 POSITIONING BENEFITS:")
        print("   🎯 Tests model attention across full context window")
        print("   📊 Enables position bias analysis")
        print("   🔍 Validates consistent retrieval performance")
        print("   ⚡ Identifies optimal placement strategies")
        print("   🧠 Reveals attention patterns and context utilization")
        print("   📈 Quantifies performance degradation by position")

    def handle_interruption(self, experiment_id: str):
        """Handle graceful shutdown with rich display."""
        # Stop the rich progress display
        if self.rich_progress:
            self.rich_progress.stop()

        # Create interruption title panel
        interruption_title = Panel(
            Align.center(Text("⚠️ EXPERIMENT INTERRUPTED", style="bold yellow")), style="yellow", box=box.DOUBLE
        )
        self.console.print(interruption_title)

        elapsed_minutes = self.progress.elapsed_time / 60

        # Create progress saved table
        progress_table = Table(title="📊 Progress Saved", box=box.ROUNDED, title_style="bold yellow")
        progress_table.add_column("Metric", style="cyan")
        progress_table.add_column("Value", style="white")

        progress_table.add_row(
            "Completed",
            f"[yellow]{self.progress.completed_total:,}/{self.progress.total_experiments:,}[/yellow] ([bold green]{self.progress.completion_percentage:.1f}%[/bold green])",
        )
        progress_table.add_row("Duration", f"[cyan]{elapsed_minutes:.1f} minutes[/cyan]")
        progress_table.add_row("Cost", f"[green]${self.progress.actual_cost:.2f}[/green]")

        self.console.print(progress_table)

        # Create resume command panel
        resume_content = f"[bold cyan]python context_advantage_experiment.py --resume-from {experiment_id}[/bold cyan]"
        resume_panel = Panel(resume_content, title="🔄 Resume Command", box=box.ROUNDED, style="blue")
        self.console.print(resume_panel)


@dataclass
class ExperimentConfig:
    """Configuration for a context advantage experiment."""

    experiment_id: str
    experiment_type: str  # 'needle', 'longmemeval', 'both'
    models: list[str]
    context_sizes: list[int]
    iterations: int

    # Needle experiment specific
    needle_compositions: list[str]  # ['pg_heavy', 'arxiv_heavy', 'mixed']
    needles_per_haystack: int

    # LongMemEval specific
    question_groups: list[str]  # ['short', 'medium', 'long', 'extended']

    # General
    output_dir: Path
    save_intermediate: bool
    max_concurrent: int


@dataclass
class ExperimentCheckpoint:
    """Checkpoint data for resuming experiments."""

    experiment_id: str
    config: ExperimentConfig

    # Completed results
    completed_needle_results: list[dict]
    completed_longmem_results: list[dict]

    # Remaining tasks
    remaining_needle_tasks: list[tuple]  # (model, context_size, composition, iteration)
    remaining_longmem_tasks: list[tuple]  # (model, group_name)

    # Progress tracking
    current_phase: str  # 'needle', 'longmemeval', 'analysis', 'complete'
    completed_tasks: int
    total_tasks: int

    # Metadata
    timestamp: float
    last_saved: float

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentCheckpoint":
        """Create checkpoint from dictionary."""
        # Convert config back to dataclass
        config_data = data["config"]
        config_data["output_dir"] = Path(config_data["output_dir"])
        config = ExperimentConfig(**config_data)

        return cls(
            experiment_id=data["experiment_id"],
            config=config,
            completed_needle_results=data["completed_needle_results"],
            completed_longmem_results=data["completed_longmem_results"],
            remaining_needle_tasks=data["remaining_needle_tasks"],
            remaining_longmem_tasks=data["remaining_longmem_tasks"],
            current_phase=data["current_phase"],
            completed_tasks=data["completed_tasks"],
            total_tasks=data["total_tasks"],
            timestamp=data["timestamp"],
            last_saved=data["last_saved"],
        )

    def to_dict(self) -> dict:
        """Convert checkpoint to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "config": {**asdict(self.config), "output_dir": str(self.config.output_dir)},
            "completed_needle_results": self.completed_needle_results,
            "completed_longmem_results": self.completed_longmem_results,
            "remaining_needle_tasks": self.remaining_needle_tasks,
            "remaining_longmem_tasks": self.remaining_longmem_tasks,
            "current_phase": self.current_phase,
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "timestamp": self.timestamp,
            "last_saved": self.last_saved,
        }


@dataclass
class NeedleExperimentResult:
    """Result from a single needle experiment."""

    experiment_id: str
    model_name: str
    context_size: int
    composition: str
    haystack_id: str
    needle_set_id: str

    # Results
    retrieval_scores: list[float]
    cross_reference_scores: list[float]
    position_analysis: dict[str, Any]

    # Position tracking
    needle_positions: list[NeedlePosition]
    question_mappings: list[QuestionNeedleMapping]

    # Performance
    avg_response_time: float
    total_cost: float
    success_rate: float

    # Metadata
    timestamp: float


@dataclass
class LongMemEvalExperimentResult:
    """Result from a single LongMemEval experiment."""

    experiment_id: str
    model_name: str
    context_group: str

    # Results
    question_scores: list[float]
    avg_score_by_type: dict[str, float]
    context_correlation: float

    # Performance
    avg_response_time: float
    total_cost: float
    success_rate: float

    # Metadata
    timestamp: float
    num_questions: int


class ContextAdvantageExperiment:
    """Main experiment coordinator for context advantage testing."""

    def __init__(self, data_dir: Path | None = None, api_key: str | None = None):
        """Initialize the experiment coordinator."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent.parent.parent / "data"

        self.data_dir = Path(data_dir)
        self.experiment_id = str(uuid.uuid4())[:8]
        api_key = api_key or os.getenv("ORQ_API_KEY")
        print(f"🔑 API Key: {api_key}")

        print(f"🧪 Initializing Context Advantage Experiment: {self.experiment_id}")

        # Initialize components
        self.model_interface = ModelInterface(api_key=api_key)
        self.haystack_builder = HaystackBuilder(data_dir=self.data_dir)
        self.needle_generator = NeedleGenerator()
        self.needle_evaluator = NeedleEvaluator()
        self.longmem_evaluator = LongMemEvalEvaluator()
        self.context_grouper = LongMemEvalContextGrouper(data_dir=self.data_dir)

        # Storage for results
        self.needle_results: list[NeedleExperimentResult] = []
        self.longmem_results: list[LongMemEvalExperimentResult] = []

        # Progress tracking
        self.progress: ExperimentProgress = None
        self.progress_tracker: ProgressTracker = None

        # Checkpoint management
        self.checkpoint_dir = self.data_dir / "checkpoints" / "context_advantage"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = 300  # Save every 5 minutes
        self.last_checkpoint_time = time.time()
        self.graceful_shutdown = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("✅ Experiment coordinator initialized")
        print(f"💾 Checkpoint directory: {self.checkpoint_dir}")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        if self.progress_tracker:
            self.progress_tracker.handle_interruption(self.experiment_id)
        else:
            print(f"\n⚠️  Received signal {signum}. Initiating graceful shutdown...")
        self.graceful_shutdown = True

    def _calculate_experiment_statistics(self, config: ExperimentConfig) -> ExperimentProgress:
        """
        Calculate total experiments, LLM calls, and cost estimates.

        This method provides detailed calculations for experiment planning and resource estimation.

        Calculation Logic:

        1. NEEDLE EXPERIMENTS:
           - Total experiments = models × context_sizes × compositions × iterations
           - Each experiment creates one haystack with N needles inserted
           - Each needle gets 2 questions (direct retrieval + cross-reference/synthesis)
           - LLM calls per experiment = needles_per_haystack × 2 questions_per_needle

        2. LONGMEMEVAL EXPERIMENTS:
           - Total experiments = models × question_groups
           - Each experiment tests one model on one question group (short/medium/long/extended)
           - Sample size limited to 50 questions per group to manage cost/time
           - LLM calls per experiment = 50 questions (fixed sample size)

        3. COST & TIME ESTIMATES:
           - Estimated cost: $0.01 per LLM call (varies by model and context length)
           - Estimated time: 3 seconds per call (includes API latency + processing)
        """

        # Calculate needle experiments
        total_needle_experiments = 0
        needle_llm_calls = 0
        if config.experiment_type in ["needle", "both"]:
            # Needle experiment calculation:
            # Each combination of (model, context_size, composition) is run multiple times (iterations)
            total_needle_experiments = (
                len(config.models) * len(config.context_sizes) * len(config.needle_compositions) * config.iterations
            )

            # Each needle experiment has questions_per_needle * needles_per_haystack LLM calls
            # Default: 2 questions per needle (1 direct retrieval + 1 cross-reference/synthesis)
            questions_per_needle = 2
            calls_per_experiment = config.needles_per_haystack * questions_per_needle
            needle_llm_calls = total_needle_experiments * calls_per_experiment

            print(f"📊 Needle Experiment Calculation:")
            print(f"   • Models: {len(config.models)}")
            print(f"   • Context sizes: {len(config.context_sizes)}")
            print(f"   • Compositions: {len(config.needle_compositions)}")
            print(f"   • Iterations: {config.iterations}")
            print(f"   • Needles per haystack: {config.needles_per_haystack}")
            print(f"   • Questions per needle: {questions_per_needle}")
            print(f"   ")
            print(f"   🧮 Calculation Steps:")
            print(
                f"   → Total experiments = {len(config.models)} models × {len(config.context_sizes)} sizes × {len(config.needle_compositions)} compositions × {config.iterations} iterations"
            )
            print(f"   → Total experiments = {total_needle_experiments:,}")
            print(
                f"   → Calls per experiment = {config.needles_per_haystack} needles × {questions_per_needle} questions/needle = {calls_per_experiment}"
            )
            print(
                f"   → Total LLM calls = {total_needle_experiments:,} experiments × {calls_per_experiment} calls/experiment = {needle_llm_calls:,}"
            )

        # Calculate longmemeval experiments
        total_longmem_experiments = 0
        longmem_llm_calls = 0
        if config.experiment_type in ["longmemeval", "both"]:
            # LongMemEval experiment calculation:
            # Each model is tested on each question group (short/medium/long/extended)
            total_longmem_experiments = len(config.models) * len(config.question_groups)

            # Each longmemeval experiment has ~50 questions (sample size limit to manage cost/time)
            questions_per_group = 50
            longmem_llm_calls = total_longmem_experiments * questions_per_group

            print(f"📊 LongMemEval Experiment Calculation:")
            print(f"   • Models: {len(config.models)}")
            print(f"   • Question groups: {len(config.question_groups)}")
            print(f"   • Sample size per group: {questions_per_group}")
            print(f"   ")
            print(f"   🧮 Calculation Steps:")
            print(
                f"   → Total experiments = {len(config.models)} models × {len(config.question_groups)} groups = {total_longmem_experiments:,}"
            )
            print(f"   → Calls per experiment = {questions_per_group} questions/group (sample size limit)")
            print(
                f"   → Total LLM calls = {total_longmem_experiments:,} experiments × {questions_per_group} calls/experiment = {longmem_llm_calls:,}"
            )

        total_experiments = total_needle_experiments + total_longmem_experiments
        total_llm_calls = needle_llm_calls + longmem_llm_calls

        # Cost estimates (rough estimates based on typical model costs)
        # Note: Actual costs vary significantly by model and context length
        estimated_cost_per_call = 0.01  # Conservative estimate - could be $0.001-$0.10+ per call
        estimated_total_cost = total_llm_calls * estimated_cost_per_call

        # Time estimates (rough estimates based on API response times)
        # Includes API latency, model inference time, and local processing
        estimated_seconds_per_call = 3.0  # Conservative estimate - could be 1-10+ seconds
        estimated_duration_minutes = (total_llm_calls * estimated_seconds_per_call) / 60

        print(f"📊 Combined Experiment Summary:")
        print(f"   🧮 Final Calculations:")
        print(
            f"   → Total experiments = {total_needle_experiments:,} needle + {total_longmem_experiments:,} longmemeval = {total_experiments:,}"
        )
        print(
            f"   → Total LLM calls = {needle_llm_calls:,} needle + {longmem_llm_calls:,} longmemeval = {total_llm_calls:,}"
        )
        print(
            f"   → Estimated cost = {total_llm_calls:,} calls × ${estimated_cost_per_call:.3f}/call = ${estimated_total_cost:.2f}"
        )
        print(
            f"   → Estimated duration = {total_llm_calls:,} calls × {estimated_seconds_per_call:.1f}s/call ÷ 60 = {estimated_duration_minutes:.1f} minutes"
        )

        return ExperimentProgress(
            total_needle_experiments=total_needle_experiments,
            total_longmem_experiments=total_longmem_experiments,
            total_experiments=total_experiments,
            total_llm_calls=total_llm_calls,
            estimated_total_cost=estimated_total_cost,
            estimated_duration_minutes=estimated_duration_minutes,
        )

    def _initialize_progress_tracking(self, config: ExperimentConfig):
        """Initialize progress tracking for the experiment."""
        self.progress = self._calculate_experiment_statistics(config)
        self.progress_tracker = ProgressTracker(self.progress)
        self.progress_tracker.start_experiment(config)

    def _should_checkpoint(self) -> bool:
        """Check if it's time to save a checkpoint."""
        return time.time() - self.last_checkpoint_time >= self.checkpoint_interval

    def _generate_checkpoint_path(self, experiment_id: str) -> Path:
        """Generate path for checkpoint file."""
        return self.checkpoint_dir / f"{experiment_id}_checkpoint.json"

    def save_results_checkpoint(self, config: ExperimentConfig) -> Path:
        """Save simplified checkpoint with just results as JSON list after each run."""
        results_dir = config.output_dir / "checkpoints"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create separate checkpoint files for needle and longmem results
        needle_checkpoint = results_dir / f"{config.experiment_id}_needle_results.json"
        longmem_checkpoint = results_dir / f"{config.experiment_id}_longmem_results.json"

        # Save needle results as list of dicts
        if self.needle_results:
            needle_data = [asdict(r) for r in self.needle_results]
            with open(needle_checkpoint, "w") as f:
                json.dump(needle_data, f, indent=2)
            print(f"💾 Saved {len(self.needle_results)} needle results to checkpoint")

        # Save longmem results as list of dicts
        if self.longmem_results:
            longmem_data = [asdict(r) for r in self.longmem_results]
            with open(longmem_checkpoint, "w") as f:
                json.dump(longmem_data, f, indent=2)
            print(f"💾 Saved {len(self.longmem_results)} LongMemEval results to checkpoint")

        # Also save a summary file
        summary_file = results_dir / f"{config.experiment_id}_summary.json"
        summary = {
            "experiment_id": config.experiment_id,
            "timestamp": time.time(),
            "needle_results_count": len(self.needle_results),
            "longmem_results_count": len(self.longmem_results),
            "config": {**asdict(config), "output_dir": str(config.output_dir)},
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        return results_dir

    def load_results_checkpoint(self, config: ExperimentConfig) -> tuple[int, int]:
        """Load results from checkpoint files."""
        results_dir = config.output_dir / "checkpoints"

        # Load needle results
        needle_checkpoint = results_dir / f"{config.experiment_id}_needle_results.json"
        if needle_checkpoint.exists():
            with open(needle_checkpoint) as f:
                needle_data = json.load(f)
            self.needle_results = [NeedleExperimentResult(**r) for r in needle_data]
            print(f"📂 Loaded {len(self.needle_results)} needle results from checkpoint")

        # Load longmem results
        longmem_checkpoint = results_dir / f"{config.experiment_id}_longmem_results.json"
        if longmem_checkpoint.exists():
            with open(longmem_checkpoint) as f:
                longmem_data = json.load(f)
            self.longmem_results = [LongMemEvalExperimentResult(**r) for r in longmem_data]
            print(f"📂 Loaded {len(self.longmem_results)} LongMemEval results from checkpoint")

        return len(self.needle_results), len(self.longmem_results)

    def save_checkpoint(
        self,
        config: ExperimentConfig,
        remaining_needle_tasks: list[tuple] = None,
        remaining_longmem_tasks: list[tuple] = None,
        current_phase: str = "needle",
        completed_tasks: int = 0,
        total_tasks: int = 0,
    ) -> Path:
        """Save experiment checkpoint."""
        checkpoint = ExperimentCheckpoint(
            experiment_id=config.experiment_id,
            config=config,
            completed_needle_results=[asdict(r) for r in self.needle_results],
            completed_longmem_results=[asdict(r) for r in self.longmem_results],
            remaining_needle_tasks=remaining_needle_tasks or [],
            remaining_longmem_tasks=remaining_longmem_tasks or [],
            current_phase=current_phase,
            completed_tasks=completed_tasks,
            total_tasks=total_tasks,
            timestamp=time.time(),
            last_saved=time.time(),
        )

        checkpoint_path = self._generate_checkpoint_path(config.experiment_id)

        # Atomic write using temporary file
        temp_path = checkpoint_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)

            # Atomic move
            temp_path.rename(checkpoint_path)

            self.last_checkpoint_time = time.time()
            print(f"💾 Checkpoint saved: {checkpoint_path}")

            return checkpoint_path

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_checkpoint(self, experiment_id: str) -> ExperimentCheckpoint:
        """Load experiment checkpoint."""
        checkpoint_path = self._generate_checkpoint_path(experiment_id)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            checkpoint = ExperimentCheckpoint.from_dict(checkpoint_data)

            # Restore results
            self.needle_results = [NeedleExperimentResult(**r) for r in checkpoint.completed_needle_results]
            self.longmem_results = [LongMemEvalExperimentResult(**r) for r in checkpoint.completed_longmem_results]

            # Update experiment ID to match checkpoint
            self.experiment_id = checkpoint.experiment_id

            print(f"📂 Checkpoint loaded: {checkpoint_path}")
            print(f"✅ Restored {len(self.needle_results)} needle + {len(self.longmem_results)} longmemeval results")
            print(
                f"📋 Phase: {checkpoint.current_phase}, Progress: {checkpoint.completed_tasks}/{checkpoint.total_tasks}"
            )

            return checkpoint

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            raise

    def list_checkpoints(self) -> list[dict]:
        """List available experiment checkpoints."""
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)

                checkpoints.append(
                    {
                        "experiment_id": data["experiment_id"],
                        "timestamp": data["timestamp"],
                        "current_phase": data["current_phase"],
                        "completed_tasks": data["completed_tasks"],
                        "total_tasks": data["total_tasks"],
                        "progress_pct": (data["completed_tasks"] / data["total_tasks"] * 100)
                        if data["total_tasks"] > 0
                        else 0,
                        "file": checkpoint_file,
                        "config": data["config"],
                    }
                )

            except Exception as e:
                print(f"⚠️  Error reading checkpoint {checkpoint_file}: {e}")

        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)

    def cleanup_checkpoint(self, experiment_id: str) -> None:
        """Clean up checkpoint file after successful completion."""
        checkpoint_path = self._generate_checkpoint_path(experiment_id)

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                print(f"🗑️  Cleaned up checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up checkpoint: {e}")

    def run_needle_experiments(
        self,
        config: ExperimentConfig,
        progress_callback: Callable | None = None,
        remaining_tasks: list[tuple] | None = None,
    ) -> list[NeedleExperimentResult]:
        """Run needle-in-haystack experiments with checkpoint recovery."""
        # Generate all tasks or use provided remaining tasks
        if remaining_tasks is None or len(remaining_tasks) == 0:
            # Generate all tasks and filter out completed ones
            all_tasks = []
            completed_combinations = set()

            # Track which combinations are already completed
            for result in self.needle_results:
                # We need to determine iteration number from existing results
                # For now, we'll use a simpler approach - just check model/context/composition
                completed_combinations.add((result.model_name, result.context_size, result.composition))

            # Generate remaining tasks
            for model_name in config.models:
                for context_size in config.context_sizes:
                    for composition in config.needle_compositions:
                        for iteration in range(config.iterations):
                            task = (model_name, context_size, composition, iteration)
                            # For needle experiments, we need a more sophisticated approach
                            # to track iterations, but for now let's just generate all remaining
                            base_combo = (model_name, context_size, composition)
                            combo_count = sum(1 for combo in completed_combinations if combo == base_combo)
                            if combo_count < config.iterations:
                                all_tasks.append(task)

            print(f"🔄 Generated {len(all_tasks)} remaining needle tasks from {len(self.needle_results)} completed")
        else:
            all_tasks = remaining_tasks.copy()

        # Initialize phase progress tracking
        if self.progress_tracker:
            self.progress_tracker.start_phase("needle", len(all_tasks))

        results = self.needle_results.copy()
        total_experiments = len(results) + len(all_tasks)
        completed = len(results)

        # Process remaining tasks - create a working copy for safe removal
        remaining_tasks_list = all_tasks.copy()

        print(f"🔄 Processing {len(all_tasks)} needle experiments (resuming from {completed} completed)")
        if remaining_tasks is not None:
            print(f"📋 Loaded remaining tasks from checkpoint: {len(remaining_tasks)} tasks")

        try:
            while all_tasks and not self.graceful_shutdown:
                current_task = all_tasks.pop(0)
                model_name, context_size, composition, iteration = current_task

                # Safely remove from remaining tasks list if it exists
                if current_task in remaining_tasks_list:
                    remaining_tasks_list.remove(current_task)
                try:
                    # Start fancy progress tracking for this experiment
                    context_display = f"{context_size // 1000}k" if context_size >= 1000 else str(context_size)
                    experiment_desc = f"{model_name} • {context_display} • {composition} • iter {iteration + 1}"
                    if self.progress_tracker:
                        self.progress_tracker.start_experiment_task(experiment_desc)

                    # Build haystack
                    if self.progress_tracker:
                        self.progress_tracker.update_experiment_task("Building haystack...", 0.1)
                    haystack = self.haystack_builder.build_haystack(
                        target_size=context_size, composition=composition, shuffle=True
                    )

                    # Generate needles
                    if self.progress_tracker:
                        self.progress_tracker.update_experiment_task("Generating needles...", 0.2)
                    needle_set = self.needle_generator.generate_needle_set(
                        num_needles=config.needles_per_haystack, domains=["pg", "arxiv"], questions_per_needle=2
                    )

                    # Insert needles into haystack with position tracking
                    if self.progress_tracker:
                        self.progress_tracker.update_experiment_task("Inserting needles...", 0.3)
                    enhanced_haystack, needle_positions, question_mappings = self._insert_needles_into_haystack(
                        haystack, needle_set
                    )

                    # Run model queries
                    if self.progress_tracker:
                        self.progress_tracker.update_experiment_task(f"Querying {model_name}...", 0.4)
                    query_results = self._query_model_with_needles(model_name, enhanced_haystack, needle_set)

                    # Evaluate results with position data
                    if self.progress_tracker:
                        self.progress_tracker.update_experiment_task("Evaluating results...", 0.9)
                    result = self._evaluate_needle_experiment(
                        model_name,
                        context_size,
                        composition,
                        haystack,
                        needle_set,
                        query_results,
                        needle_positions,
                        question_mappings,
                    )

                    # Complete the experiment task
                    if self.progress_tracker:
                        self.progress_tracker.complete_experiment_task()

                    results.append(result)
                    self.needle_results.append(result)
                    completed += 1

                    # Update progress tracking
                    if self.progress_tracker:
                        self.progress_tracker.update_experiment_progress(result, "needle")

                    if progress_callback:
                        progress_callback(completed, total_experiments, result)

                    # Save intermediate results
                    if config.save_intermediate:
                        self._save_intermediate_result(result, config.output_dir)

                    # Save both types of checkpoints after EACH run completes
                    # 1. Simple results checkpoint (list of dicts as JSON)
                    self.save_results_checkpoint(config)

                    # 2. Full checkpoint with task tracking (less frequent)
                    if self._should_checkpoint():
                        self.save_checkpoint(
                            config=config,
                            remaining_needle_tasks=remaining_tasks_list,
                            current_phase="needle",
                            completed_tasks=completed,
                            total_tasks=total_experiments,
                        )

                except Exception as e:
                    print(f"❌ Error in needle experiment: {e}")
                    completed += 1
                    # Save checkpoint even on error
                    self.save_checkpoint(
                        config=config,
                        remaining_needle_tasks=remaining_tasks_list,
                        current_phase="needle",
                        completed_tasks=completed,
                        total_tasks=total_experiments,
                    )
                    continue

        except KeyboardInterrupt:
            print("\n⚠️  Needle experiments interrupted by user")
            self.graceful_shutdown = True

        # Save final checkpoint before finishing phase
        if remaining_tasks_list or self.graceful_shutdown:
            self.save_checkpoint(
                config=config,
                remaining_needle_tasks=remaining_tasks_list,
                current_phase="needle_interrupted" if self.graceful_shutdown else "needle_complete",
                completed_tasks=completed,
                total_tasks=total_experiments,
            )

            if self.graceful_shutdown:
                print(f"💾 Progress saved. Resume with experiment ID: {config.experiment_id}")
                return results

        # Display phase completion summary
        if self.progress_tracker:
            self.progress_tracker.display_phase_summary("needle", results)

        return results

    def run_longmemeval_experiments(
        self,
        config: ExperimentConfig,
        progress_callback: Callable | None = None,
        remaining_tasks: list[tuple] | None = None,
    ) -> list[LongMemEvalExperimentResult]:
        """Run LongMemEval experiments with checkpoint recovery."""

        # Load and group LongMemEval data
        try:
            questions_data = self.context_grouper.load_longmemeval_data()
            processed_questions = self.context_grouper.process_questions(questions_data)
            context_groups = self.context_grouper.create_context_groups(processed_questions)
        except Exception as e:
            print(f"❌ Error loading LongMemEval data: {e}")
            print("Skipping LongMemEval experiments")
            return []

        results = self.longmem_results.copy()

        # Generate all tasks or use provided remaining tasks
        if remaining_tasks is None or len(remaining_tasks) == 0:
            # Generate all tasks and filter out completed ones
            all_tasks = []
            completed_combinations = set()

            # Track which combinations are already completed
            for result in self.longmem_results:
                completed_combinations.add((result.model_name, result.context_group))

            # Generate remaining tasks
            for model_name in config.models:
                for group_name in config.question_groups:
                    task = (model_name, group_name)
                    if task not in completed_combinations:
                        all_tasks.append(task)

            print(
                f"🔄 Generated {len(all_tasks)} remaining LongMemEval tasks from {len(completed_combinations)} completed"
            )
        else:
            all_tasks = remaining_tasks.copy()

        # Initialize phase progress tracking
        if self.progress_tracker:
            self.progress_tracker.start_phase("longmemeval", len(all_tasks))

        total_experiments = len(results) + len(all_tasks)
        completed = len(results)

        # Process remaining tasks - create a working copy for safe removal
        remaining_tasks_list = all_tasks.copy()

        print(f"🔄 Processing {len(all_tasks)} LongMemEval experiments (resuming from {completed} completed)")
        if remaining_tasks is not None:
            print(f"📋 Loaded remaining tasks from checkpoint: {len(remaining_tasks)} tasks")

        try:
            while all_tasks and not self.graceful_shutdown:
                current_task = all_tasks.pop(0)
                model_name, group_name = current_task

                # Safely remove from remaining tasks list if it exists
                if current_task in remaining_tasks_list:
                    remaining_tasks_list.remove(current_task)
                try:
                    # Start fancy progress tracking for this experiment
                    experiment_desc = f"{model_name} • {group_name} group"
                    if self.progress_tracker:
                        self.progress_tracker.start_experiment_task(experiment_desc)

                    if group_name not in context_groups:
                        if self.progress_tracker:
                            self.progress_tracker.update_experiment_task(f"⚠️ Group {group_name} not found")
                            self.progress_tracker.complete_experiment_task()
                        completed += 1
                        continue

                    group = context_groups[group_name]
                    if not group.questions:
                        if self.progress_tracker:
                            self.progress_tracker.update_experiment_task(f"⚠️ No questions in {group_name}")
                            self.progress_tracker.complete_experiment_task()
                        completed += 1
                        continue

                    # Sample questions if there are too many
                    if self.progress_tracker:
                        self.progress_tracker.update_experiment_task("Sampling questions...", 0.1)
                    sample_size = min(50, len(group.questions))  # Limit to 50 questions per group
                    sampled_questions = group.questions[:sample_size]

                    # Run model queries
                    if self.progress_tracker:
                        self.progress_tracker.update_experiment_task(
                            f"Querying {model_name} ({sample_size} questions)...", 0.2
                        )
                    query_results = self._query_model_with_longmemeval(model_name, sampled_questions)

                    # Evaluate results
                    if self.progress_tracker:
                        self.progress_tracker.update_experiment_task("Evaluating results...", 0.9)
                    result = self._evaluate_longmemeval_experiment(
                        model_name, group_name, sampled_questions, query_results
                    )

                    # Complete the experiment task
                    if self.progress_tracker:
                        self.progress_tracker.complete_experiment_task()

                    results.append(result)
                    self.longmem_results.append(result)
                    completed += 1

                    # Update progress tracking
                    if self.progress_tracker:
                        self.progress_tracker.update_experiment_progress(result, "longmemeval")

                    if progress_callback:
                        progress_callback(completed, total_experiments, result)

                    # Save intermediate results
                    if config.save_intermediate:
                        self._save_intermediate_result(result, config.output_dir)

                    # Save both types of checkpoints after EACH run completes
                    # 1. Simple results checkpoint (list of dicts as JSON)
                    self.save_results_checkpoint(config)

                    # 2. Full checkpoint with task tracking (less frequent)
                    if self._should_checkpoint():
                        self.save_checkpoint(
                            config=config,
                            remaining_longmem_tasks=remaining_tasks_list,
                            current_phase="longmemeval",
                            completed_tasks=completed,
                            total_tasks=total_experiments,
                        )

                except Exception as e:
                    print(f"❌ Error in LongMemEval experiment: {e}")
                    completed += 1
                    # Save checkpoint even on error
                    self.save_checkpoint(
                        config=config,
                        remaining_longmem_tasks=remaining_tasks_list,
                        current_phase="longmemeval",
                        completed_tasks=completed,
                        total_tasks=total_experiments,
                    )
                    continue

        except KeyboardInterrupt:
            print("\n⚠️  LongMemEval experiments interrupted by user")
            self.graceful_shutdown = True

        # Save final checkpoint before finishing phase
        if remaining_tasks_list or self.graceful_shutdown:
            self.save_checkpoint(
                config=config,
                remaining_longmem_tasks=remaining_tasks_list,
                current_phase="longmemeval_interrupted" if self.graceful_shutdown else "longmemeval_complete",
                completed_tasks=completed,
                total_tasks=total_experiments,
            )

            if self.graceful_shutdown:
                print(f"💾 Progress saved. Resume with experiment ID: {config.experiment_id}")
                return results

        # Display phase completion summary
        if self.progress_tracker:
            self.progress_tracker.display_phase_summary("longmemeval", results)

        return results

    def _insert_needles_into_haystack(
        self, haystack: Haystack, needle_set: NeedleSet
    ) -> tuple[str, list[NeedlePosition], list[QuestionNeedleMapping]]:
        """Insert needles into haystack at strategic positions with precise tracking."""
        import tiktoken

        # Initialize tokenizer for position calculation
        encoding = tiktoken.encoding_for_model("gpt-4o")

        # Split haystack into sections
        sections = haystack.content.split("\n\n---\n\n")
        separator = "\n\n---\n\n"

        # Define insertion positions and distribute needles
        positions = ["beginning", "early_middle", "center", "late_middle", "end"]
        needles_per_zone = len(needle_set.needles) // len(positions)
        remaining_needles = len(needle_set.needles) % len(positions)

        # Calculate needles per zone with remainder distribution
        zone_needle_counts = [needles_per_zone] * len(positions)
        for i in range(remaining_needles):
            zone_needle_counts[i] += 1

        needle_positions = []
        question_mappings = []
        enhanced_sections = []
        current_char_pos = 0
        current_token_pos = 0  # Track cumulative token position
        needle_idx = 0

        # Track which needles go in which zones
        zone_assignments = {}
        needle_zone_idx = 0
        for zone_idx, count in enumerate(zone_needle_counts):
            zone_assignments[positions[zone_idx]] = needle_set.needles[needle_zone_idx : needle_zone_idx + count]
            needle_zone_idx += count

        for section_idx, section in enumerate(sections):
            # Determine zone for this section
            section_ratio = section_idx / max(len(sections) - 1, 1)  # Avoid division by zero

            current_zone = None
            if section_ratio <= 0.1:
                current_zone = "beginning"
            elif section_ratio <= 0.35:
                current_zone = "early_middle"
            elif section_ratio <= 0.65:
                current_zone = "center"
            elif section_ratio <= 0.85:
                current_zone = "late_middle"
            else:
                current_zone = "end"

            # Start with original section
            enhanced_section = section
            section_start_pos = current_char_pos
            section_start_token = current_token_pos

            # Update token position for this section
            section_tokens = len(encoding.encode(section))
            current_token_pos += section_tokens

            # Insert needles assigned to this zone
            needles_for_zone = zone_assignments.get(current_zone, [])

            if needles_for_zone and section_idx % max(1, len(sections) // len(positions)) == 0:
                # Insert the next needle from this zone
                if needle_idx < len(needle_set.needles):
                    needle = needle_set.needles[needle_idx]

                    # Calculate precise position
                    needle_content = f"\n\n{needle.content}"
                    char_start = current_char_pos + len(enhanced_section)
                    char_end = char_start + len(needle_content)

                    # Calculate token positions correctly - use cumulative position
                    token_start = current_token_pos  # Position after current section
                    needle_tokens = len(encoding.encode(needle_content))
                    token_end = token_start + needle_tokens

                    # Update cumulative token position to include needle
                    current_token_pos += needle_tokens

                    # Create position record
                    position = NeedlePosition(
                        needle_id=needle.id,
                        section_index=section_idx,
                        section_ratio=section_ratio,
                        position_zone=current_zone,
                        char_start=char_start,
                        char_end=char_end,
                        token_start=token_start,
                        token_end=token_end,
                    )

                    needle_positions.append(position)

                    # Add needle to section
                    enhanced_section += needle_content
                    needle_idx += 1

            enhanced_sections.append(enhanced_section)
            current_char_pos += len(enhanced_section)

            # Add separator length if not last section
            if section_idx < len(sections) - 1:
                current_char_pos += len(separator)
                # Also add separator tokens to cumulative count
                separator_tokens = len(encoding.encode(separator))
                current_token_pos += separator_tokens

        # Create question-to-needle mappings
        for question in needle_set.questions:
            required_needle_ids = question.required_needles

            # Find positions for required needles
            primary_position = None
            secondary_positions = []

            for needle_id in required_needle_ids:
                position = next((p for p in needle_positions if p.needle_id == needle_id), None)
                if position:
                    if primary_position is None:
                        primary_position = position
                    else:
                        secondary_positions.append(position)

            if primary_position:
                mapping = QuestionNeedleMapping(
                    question_id=question.id,
                    needle_ids=required_needle_ids,
                    primary_position=primary_position,
                    secondary_positions=secondary_positions,
                )
                question_mappings.append(mapping)

        # Update haystack object with positions
        haystack.needle_positions = [p.char_start for p in needle_positions]

        enhanced_content = separator.join(enhanced_sections)

        # Verify token positions with final content
        final_total_tokens = len(encoding.encode(enhanced_content))

        print(f"📍 Inserted {len(needle_positions)} needles with precise position tracking")
        print(f"📏 Document stats: {len(enhanced_content):,} characters, ~{final_total_tokens:,} tokens")

        for pos in needle_positions:
            # Calculate percentage through document for verification
            char_percentage = (pos.char_start / len(enhanced_content)) * 100 if enhanced_content else 0
            token_percentage = (pos.token_start / final_total_tokens) * 100 if final_total_tokens else 0

            print(
                f"   Needle {pos.needle_id}: {pos.position_zone} "
                f"(char {pos.char_start}-{pos.char_end} = {char_percentage:.1f}%, "
                f"token {pos.token_start}-{pos.token_end} = {token_percentage:.1f}%)"
            )

        return enhanced_content, needle_positions, question_mappings

    def _query_model_with_needles(
        self, model_name: str, haystack_content: str, needle_set: NeedleSet
    ) -> list[QueryResult]:
        """Query model with needle questions."""
        results = []

        for question in needle_set.questions:
            prompt = f"Based on the following context, please answer the question.\n\nContext:\n{haystack_content}\n\nQuestion: {question.question}\n\nAnswer:"

            result = self.model_interface.query_model(
                model_name=model_name, prompt=prompt, max_tokens=500, temperature=0.0
            )

            results.append(result)
            time.sleep(0.5)  # Pause between requests to avoid rate limits

        return results

    def _query_model_with_longmemeval(self, model_name: str, questions: list[Any]) -> list[QueryResult]:
        """Query model with LongMemEval questions."""
        results = []

        for question in questions:
            # Build context from question documents
            context = "\n\n".join(question.context_docs)
            prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {question.question}\n\nAnswer:"

            result = self.model_interface.query_model(
                model_name=model_name, prompt=prompt, max_tokens=1000, temperature=0.0
            )

            results.append(result)
            time.sleep(0.5)  # Pause between requests to avoid rate limits

        return results

    def _evaluate_needle_experiment(
        self,
        model_name: str,
        context_size: int,
        composition: str,
        haystack: Haystack,
        needle_set: NeedleSet,
        query_results: list[QueryResult],
        needle_positions: list[NeedlePosition],
        question_mappings: list[QuestionNeedleMapping],
    ) -> NeedleExperimentResult:
        """Evaluate results of a needle experiment."""

        retrieval_scores = []
        cross_reference_scores = []
        response_times = []
        costs = []
        successes = 0

        for i, (question, result) in enumerate(zip(needle_set.questions, query_results, strict=False)):
            if result.success:
                successes += 1

                # Evaluate based on question type
                if question.question_type == "direct":
                    score = self.needle_evaluator.evaluate_retrieval(question.expected_answer, result.response)
                    retrieval_scores.append(score.overall_score)
                elif question.question_type in ["cross_reference", "synthesis"]:
                    # Simplified cross-reference evaluation
                    score = self.needle_evaluator.evaluate_retrieval(question.expected_answer, result.response)
                    cross_reference_scores.append(score.overall_score)

            response_times.append(result.response_time)
            costs.append(result.estimated_cost)

        # Enhanced position analysis using real position data
        position_analysis = self._analyze_position_performance(
            needle_set.questions, query_results, needle_positions, question_mappings
        )

        return NeedleExperimentResult(
            experiment_id=self.experiment_id,
            model_name=model_name,
            context_size=context_size,
            composition=composition,
            haystack_id=haystack.target_size,  # Using target_size as ID
            needle_set_id=needle_set.id,
            retrieval_scores=retrieval_scores,
            cross_reference_scores=cross_reference_scores,
            position_analysis=position_analysis,
            needle_positions=needle_positions,
            question_mappings=question_mappings,
            avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
            total_cost=sum(costs),
            success_rate=successes / len(query_results) if query_results else 0,
            timestamp=time.time(),
        )

    def _analyze_position_performance(
        self,
        questions: list,
        query_results: list[QueryResult],
        needle_positions: list[NeedlePosition],
        question_mappings: list[QuestionNeedleMapping],
    ) -> dict[str, Any]:
        """Analyze performance by needle position zones."""

        # Initialize zone statistics
        zones = ["beginning", "early_middle", "center", "late_middle", "end"]
        zone_stats = {zone: {"scores": [], "questions": 0, "successes": 0} for zone in zones}

        # Create mapping from question to result score
        question_scores = {}

        for i, (question, result) in enumerate(zip(questions, query_results, strict=False)):
            if result.success:
                # Calculate score for this question
                if question.question_type == "direct" or question.question_type in ["cross_reference", "synthesis"]:
                    score = self.needle_evaluator.evaluate_retrieval(question.expected_answer, result.response)
                    score_value = score.overall_score
                else:
                    score_value = 0.5  # Default score

                question_scores[question.id] = score_value
            else:
                question_scores[question.id] = 0.0

        # Map scores to position zones
        for mapping in question_mappings:
            question_id = mapping.question_id
            primary_zone = mapping.primary_position.position_zone
            score = question_scores.get(question_id, 0.0)

            zone_stats[primary_zone]["scores"].append(score)
            zone_stats[primary_zone]["questions"] += 1
            if score > 0.5:  # Consider >0.5 as success
                zone_stats[primary_zone]["successes"] += 1

        # Calculate statistics by zone
        zone_analysis = {}
        for zone, stats in zone_stats.items():
            if stats["questions"] > 0:
                zone_analysis[zone] = {
                    "avg_score": sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0.0,
                    "success_rate": stats["successes"] / stats["questions"],
                    "total_questions": stats["questions"],
                    "needle_count": len([p for p in needle_positions if p.position_zone == zone]),
                }
            else:
                zone_analysis[zone] = {
                    "avg_score": 0.0,
                    "success_rate": 0.0,
                    "total_questions": 0,
                    "needle_count": len([p for p in needle_positions if p.position_zone == zone]),
                }

        # Overall position distribution
        position_distribution = {
            "total_needles": len(needle_positions),
            "needles_by_zone": {zone: len([p for p in needle_positions if p.position_zone == zone]) for zone in zones},
            "average_char_positions": {
                zone: sum(p.char_start for p in needle_positions if p.position_zone == zone)
                / max(len([p for p in needle_positions if p.position_zone == zone]), 1)
                for zone in zones
            },
            "average_token_positions": {
                zone: sum(p.token_start for p in needle_positions if p.position_zone == zone)
                / max(len([p for p in needle_positions if p.position_zone == zone]), 1)
                for zone in zones
            },
        }

        return {
            "zone_performance": zone_analysis,
            "position_distribution": position_distribution,
            "question_position_mappings": len(question_mappings),
            "zones_tested": len([zone for zone, stats in zone_analysis.items() if stats["total_questions"] > 0]),
        }

    def _evaluate_longmemeval_experiment(
        self, model_name: str, group_name: str, questions: list[Any], query_results: list[QueryResult]
    ) -> LongMemEvalExperimentResult:
        """Evaluate results of a LongMemEval experiment."""

        question_scores = []
        scores_by_type = defaultdict(list)
        response_times = []
        costs = []
        successes = 0

        for question, result in zip(questions, query_results, strict=False):
            if result.success:
                successes += 1

                score = self.longmem_evaluator.evaluate_answer(
                    question.answer, result.response, question.complexity.question_type
                )

                question_scores.append(score)
                scores_by_type[question.complexity.question_type].append(score)

            response_times.append(result.response_time)
            costs.append(result.estimated_cost)

        # Calculate averages by type
        avg_score_by_type = {qtype: sum(scores) / len(scores) for qtype, scores in scores_by_type.items()}

        # Simple correlation calculation (placeholder)
        context_correlation = 0.0  # Would need actual implementation

        return LongMemEvalExperimentResult(
            experiment_id=self.experiment_id,
            model_name=model_name,
            context_group=group_name,
            question_scores=question_scores,
            avg_score_by_type=avg_score_by_type,
            context_correlation=context_correlation,
            avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
            total_cost=sum(costs),
            success_rate=successes / len(query_results) if query_results else 0,
            timestamp=time.time(),
            num_questions=len(questions),
        )

    def _save_intermediate_result(self, result: Any, output_dir: Path) -> None:
        """Save intermediate result to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(result, NeedleExperimentResult):
            filename = (
                f"needle_{result.model_name}_{result.context_size}_{result.composition}_{int(result.timestamp)}.json"
            )
        elif isinstance(result, LongMemEvalExperimentResult):
            filename = f"longmem_{result.model_name}_{result.context_group}_{int(result.timestamp)}.json"
        else:
            filename = f"result_{int(time.time())}.json"

        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(asdict(result), f, indent=2)

    def analyze_results(self) -> dict[str, Any]:
        """Analyze all experimental results."""
        print("\n📊 Analyzing Experimental Results")

        analysis = {
            "experiment_id": self.experiment_id,
            "needle_experiments": len(self.needle_results),
            "longmemeval_experiments": len(self.longmem_results),
            "models_tested": [],  # Changed from set() to list
            "context_window_analysis": {},
            "cross_validation": {},
        }

        # Collect all tested models
        all_models = []
        if self.needle_results:
            all_models.extend([r.model_name for r in self.needle_results])
        if self.longmem_results:
            all_models.extend([r.model_name for r in self.longmem_results])
        analysis["models_tested"] = list(set(all_models))  # Remove duplicates

        # Analyze needle results
        if self.needle_results:
            needle_analysis = self._analyze_needle_results()
            analysis.update(needle_analysis)

        # Analyze LongMemEval results
        if self.longmem_results:
            longmem_analysis = self._analyze_longmemeval_results()
            analysis.update(longmem_analysis)

        # Cross-validation analysis
        if self.needle_results and self.longmem_results:
            cross_val = self._cross_validate_results()
            analysis["cross_validation"] = cross_val

        return analysis

    def _analyze_needle_results(self) -> dict[str, Any]:
        """Analyze needle experiment results."""
        model_performance = defaultdict(lambda: defaultdict(list))

        for result in self.needle_results:
            model_performance[result.model_name][result.context_size].extend(result.retrieval_scores)

        # Calculate model capacity comparison
        high_cap_models = []
        medium_cap_models = []

        for model_name in model_performance.keys():
            config = self.model_interface.get_model_config(model_name)
            if config.max_context_tokens >= 1000000:
                if model_name not in high_cap_models:  # Avoid duplicates
                    high_cap_models.append(model_name)
            else:
                if model_name not in medium_cap_models:  # Avoid duplicates
                    medium_cap_models.append(model_name)

        return {
            "needle_analysis": {
                "model_performance": dict(model_performance),
                "high_capacity_models": high_cap_models,
                "medium_capacity_models": medium_cap_models,
            }
        }

    def _analyze_longmemeval_results(self) -> dict[str, Any]:
        """Analyze LongMemEval results."""
        group_performance = defaultdict(lambda: defaultdict(list))

        for result in self.longmem_results:
            group_performance[result.model_name][result.context_group].extend(result.question_scores)

        return {"longmemeval_analysis": {"group_performance": dict(group_performance)}}

    def _cross_validate_results(self) -> dict[str, Any]:
        """Cross-validate needle and LongMemEval results."""
        # Find common models and context sizes
        needle_models = list(set([r.model_name for r in self.needle_results]))
        longmem_models = list(set([r.model_name for r in self.longmem_results]))

        # Find intersection using list comprehension instead of set operations
        common_models = [model for model in needle_models if model in longmem_models]

        correlation_analysis = {}

        for model in common_models:
            needle_scores = [
                score for r in self.needle_results if r.model_name == model for score in r.retrieval_scores
            ]

            longmem_scores = [
                score for r in self.longmem_results if r.model_name == model for score in r.question_scores
            ]

            if needle_scores and longmem_scores:
                # Simple correlation (would use numpy.corrcoef in practice)
                avg_needle = sum(needle_scores) / len(needle_scores)
                avg_longmem = sum(longmem_scores) / len(longmem_scores)

                correlation_analysis[model] = {
                    "needle_avg": avg_needle,
                    "longmemeval_avg": avg_longmem,
                    "correlation": abs(avg_needle - avg_longmem),  # Simplified
                }

        return correlation_analysis

    def save_final_results(self, analysis: dict[str, Any], output_dir: Path) -> None:
        """Save final experimental results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save complete analysis
        analysis_path = output_dir / f"context_advantage_analysis_{self.experiment_id}.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)

        # Save raw results
        if self.needle_results:
            needle_path = output_dir / f"needle_results_{self.experiment_id}.json"
            with open(needle_path, "w") as f:
                json.dump([asdict(r) for r in self.needle_results], f, indent=2)

        if self.longmem_results:
            longmem_path = output_dir / f"longmemeval_results_{self.experiment_id}.json"
            with open(longmem_path, "w") as f:
                json.dump([asdict(r) for r in self.longmem_results], f, indent=2)

        print(f"\n💾 Final results saved to {output_dir}")
        print(f"📊 Analysis: {analysis_path}")

        return analysis_path

    def run_experiment_with_recovery(
        self, config: ExperimentConfig, resume_from: str | None = None
    ) -> tuple[list[NeedleExperimentResult], list[LongMemEvalExperimentResult]]:
        """Run experiment with automatic checkpoint recovery."""

        # Initialize progress tracking if not resuming or if resuming without progress tracker
        if not self.progress_tracker:
            self._initialize_progress_tracking(config)

        # Update progress tracker with already completed work when resuming
        if resume_from and self.progress_tracker:
            completed_so_far = len(self.needle_results) + len(self.longmem_results)
            if completed_so_far > 0:
                print(f"📊 Updating progress tracker with {completed_so_far} completed experiments")
                # Update the progress counters to reflect completed work
                self.progress.completed_needle = len(self.needle_results)
                self.progress.completed_longmem = len(self.longmem_results)
                self.progress.completed_total = completed_so_far

                # Update the main progress task to reflect completed work
                self.progress_tracker.rich_progress.update(
                    self.progress_tracker.main_task,
                    completed=completed_so_far,
                    description=f"Overall Progress (resuming) • {completed_so_far} completed",
                )

        # Handle resuming from checkpoint
        if resume_from:
            print(f"📂 Resuming experiment from checkpoint: {resume_from}")

            # First try to load the full checkpoint
            try:
                checkpoint = self.load_checkpoint(resume_from)
                config = checkpoint.config  # Use config from checkpoint

                remaining_needle_tasks = checkpoint.remaining_needle_tasks
                remaining_longmem_tasks = checkpoint.remaining_longmem_tasks
                current_phase = checkpoint.current_phase

                # Calculate expected totals for validation
                expected_needle_total = (
                    len(config.models) * len(config.context_sizes) * len(config.needle_compositions) * config.iterations
                )
                expected_longmem_total = len(config.models) * len(config.question_groups)

                # Validate checkpoint state and determine correct phase
                needle_completed = len(self.needle_results)
                longmem_completed = len(self.longmem_results)

                print(f"📊 Experiment Progress:")
                print(f"   • Needle: {needle_completed}/{expected_needle_total}")
                print(f"   • LongMemEval: {longmem_completed}/{expected_longmem_total}")

                # Determine correct phase based on actual progress
                if needle_completed < expected_needle_total:
                    current_phase = "needle"
                    print(f"🔄 Continuing needle experiments ({expected_needle_total - needle_completed} remaining)")
                elif longmem_completed < expected_longmem_total and config.experiment_type in ["longmemeval", "both"]:
                    current_phase = "longmemeval"
                    print(
                        f"🔄 Starting LongMemEval experiments ({expected_longmem_total - longmem_completed} remaining)"
                    )
                else:
                    current_phase = "complete"
                    print(f"✅ All experiments actually completed")

            except FileNotFoundError:
                # If full checkpoint doesn't exist, try loading from results checkpoints
                print("⚠️  Full checkpoint not found, attempting to load from results checkpoints...")
                config.experiment_id = resume_from
                needle_count, longmem_count = self.load_results_checkpoint(config)

                # Generate remaining tasks based on what's been completed
                remaining_needle_tasks = None  # Will regenerate based on completed count
                remaining_longmem_tasks = None
                current_phase = "needle" if needle_count == 0 else "longmemeval"
                print(f"📊 Loaded {needle_count} needle and {longmem_count} longmemeval results")
        else:
            # Also try to load any existing results checkpoints for this experiment
            self.load_results_checkpoint(config)
            remaining_needle_tasks = None
            remaining_longmem_tasks = None
            current_phase = "needle"

        # Progress tracking callback - now handled by main progress bar
        def progress_callback(completed: int, total: int, result: Any):
            # Progress is now tracked automatically by the main progress bar
            pass

        # Run needle experiments if needed
        if config.experiment_type in ["needle", "both"] and current_phase in ["needle", "needle_interrupted"]:
            try:
                needle_results = self.run_needle_experiments(config, progress_callback, remaining_needle_tasks)

                # Check if we should continue to longmemeval phase
                if self.graceful_shutdown:
                    return needle_results, []

                # After needle experiments, check if we need to run longmemeval
                if config.experiment_type == "both":
                    current_phase = "longmemeval"

            except Exception as e:
                print(f"❌ Error in needle experiments: {e}")
                # Save checkpoint before failing
                self.save_checkpoint(
                    config=config,
                    remaining_needle_tasks=remaining_needle_tasks or [],
                    current_phase="needle_error",
                    completed_tasks=len(self.needle_results),
                    total_tasks=len(config.models)
                    * len(config.context_sizes)
                    * len(config.needle_compositions)
                    * config.iterations,
                )
                raise

        # Run longmemeval experiments if needed
        if config.experiment_type in ["longmemeval", "both"] and current_phase in [
            "longmemeval",
            "longmemeval_interrupted",
        ]:
            try:
                longmem_results = self.run_longmemeval_experiments(config, progress_callback, remaining_longmem_tasks)

                if self.graceful_shutdown:
                    return self.needle_results, longmem_results

            except Exception as e:
                print(f"❌ Error in longmemeval experiments: {e}")
                # Save checkpoint before failing
                self.save_checkpoint(
                    config=config,
                    remaining_longmem_tasks=remaining_longmem_tasks or [],
                    current_phase="longmemeval_error",
                    completed_tasks=len(self.longmem_results),
                    total_tasks=len(config.models) * len(config.question_groups),
                )
                raise

        # Mark experiment as complete
        if not self.graceful_shutdown:
            self.save_checkpoint(
                config=config,
                remaining_needle_tasks=[],
                remaining_longmem_tasks=[],
                current_phase="complete",
                completed_tasks=len(self.needle_results) + len(self.longmem_results),
                total_tasks=len(self.needle_results) + len(self.longmem_results),
            )
            # Display final experiment completion
            if self.progress_tracker:
                self.progress_tracker.finish_experiment()

        return self.needle_results, self.longmem_results


def main():
    """Command line interface for running experiments."""
    parser = argparse.ArgumentParser(description="Run context window advantage experiments")
    parser.add_argument(
        "--experiment-type", choices=["needle", "longmemeval", "both"], default="both", help="Type of experiment to run"
    )
    parser.add_argument("--models", nargs="+", default=["claude-sonnet-4", "claude-haiku-3.5"], help="Models to test")
    parser.add_argument(
        "--context-sizes",
        type=str,
        default="10000,50000,100000",
        help="Comma-separated context sizes for needle experiments (optimized set)",
    )
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per configuration")
    parser.add_argument("--needles-per-haystack", type=int, default=5, help="Number of needles per haystack")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--data-dir", type=Path, help="Data directory")
    parser.add_argument(
        "--full-experiment", action="store_true", help="Run complete experiment with all models and sizes"
    )
    parser.add_argument("--resume-from", type=str, help="Resume experiment from checkpoint ID")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints and exit")
    parser.add_argument("--checkpoint-interval", type=int, default=300, help="Checkpoint save interval in seconds")
    parser.add_argument(
        "--ignore-checkpoints", action="store_true", help="Start fresh experiment ignoring any existing checkpoints"
    )

    args = parser.parse_args()

    # Initialize experiment to access checkpoint functionality
    experiment = ContextAdvantageExperiment(data_dir=args.data_dir)

    # Set checkpoint interval if provided
    if hasattr(args, "checkpoint_interval"):
        experiment.checkpoint_interval = args.checkpoint_interval

    # Handle listing checkpoints
    if args.list_checkpoints:
        checkpoints = experiment.list_checkpoints()
        if not checkpoints:
            print("📁 No checkpoints found")
            return

        print("📋 Available Checkpoints:")
        print("=" * 80)

        for checkpoint in checkpoints:
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(checkpoint["timestamp"]))
            print(f"🧪 {checkpoint['experiment_id']}")
            print(f"   Created: {timestamp_str}")
            print(f"   Phase: {checkpoint['current_phase']}")
            print(
                f"   Progress: {checkpoint['completed_tasks']}/{checkpoint['total_tasks']} ({checkpoint['progress_pct']:.1f}%)"
            )
            print(f"   Models: {checkpoint['config']['models']}")
            print(f"   Type: {checkpoint['config']['experiment_type']}")
            print()

        return

    # Parse context sizes
    context_sizes = [int(x.strip()) for x in args.context_sizes.split(",")]

    # Set defaults for full experiment
    if args.full_experiment:
        # Use model comparison pairs instead of arbitrary model list
        interface = experiment.model_interface
        all_pairs = interface.get_model_comparison_pairs()
        # Extract all unique models from pairs that exist
        args.models = list(set([model for pair in all_pairs for model in pair if model in interface.MODELS]))
        context_sizes = [10000, 50000, 100000]  # Removed 150k for efficiency
        args.iterations = 3  # Reduced from 5 for efficiency

    if args.output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent.parent
        args.output_dir = project_root / "results" / "context_advantage"

    # Create experiment config
    config = ExperimentConfig(
        experiment_id=str(uuid.uuid4())[:8],
        experiment_type=args.experiment_type,
        models=args.models,
        context_sizes=context_sizes,
        iterations=args.iterations,
        needle_compositions=["mixed"],  # Simplified to single composition for efficiency
        needles_per_haystack=args.needles_per_haystack,
        question_groups=["short", "medium", "long", "extended"],
        output_dir=args.output_dir,
        save_intermediate=True,
        max_concurrent=3,
    )

    try:
        # Test connection
        if not experiment.model_interface.test_connection():
            print("❌ Failed to connect to model API. Check your API key.")
            return

        print(f"\n🚀 Starting Context Advantage Experiment: {config.experiment_id}")
        print("📋 Configuration:")
        print(f"   Type: {config.experiment_type}")
        print(f"   Models: {config.models}")
        print(f"   Context Sizes: {config.context_sizes}")
        print(f"   Iterations: {config.iterations}")

        if args.resume_from:
            print(f"📂 Resuming from checkpoint: {args.resume_from}")

        # Run experiments with recovery
        needle_results, longmem_results = experiment.run_experiment_with_recovery(config, resume_from=args.resume_from)

        # Check if experiment was interrupted
        if experiment.graceful_shutdown:
            print("\n⚠️  Experiment interrupted gracefully")
            print(f"💾 Resume with: --resume-from {config.experiment_id}")
            return

        # Analyze results if completed
        analysis = experiment.analyze_results()

        # Save results
        results_path = experiment.save_final_results(analysis, config.output_dir)

        # Clean up checkpoint since experiment completed successfully
        experiment.cleanup_checkpoint(config.experiment_id)

        print("\n🎉 Experiment Complete!")
        print(f"📊 Total Results: {len(needle_results)} needle + {len(longmem_results)} LongMemEval")
        print(f"💾 Results saved to: {results_path}")

    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        if "config" in locals() and hasattr(experiment, "experiment_id"):
            print(f"💾 Resume with: --resume-from {experiment.experiment_id}")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        if "config" in locals() and hasattr(experiment, "experiment_id"):
            print(f"💾 Resume with: --resume-from {experiment.experiment_id}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
