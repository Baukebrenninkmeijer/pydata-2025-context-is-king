#!/usr/bin/env python3
"""
CLI Interface for Reranking Value Experiment

Usage:
    python run_experiment.py --quick-test
    python run_experiment.py --longmemeval-path data/LongMemEval/ --output-dir results/
    python run_experiment.py --resume-from experiment_id_12345
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from reranking_experiment import ExperimentConfig, RerankingValueExperiment
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def create_quick_test_config(args=None) -> ExperimentConfig:
    """Create configuration for quick testing."""
    # Use judge_model from args if provided, otherwise use default
    judge_model = getattr(args, 'judge_model', 'gpt-4o-mini') if args else 'gpt-4o-mini'
    
    return ExperimentConfig(
        experiment_id=f"quick_test_{int(time.time())}",
        output_dir="results/quick_test",
        longmemeval_path="data/",
        document_collections=["test"],
        models=["claude-sonnet-3.7"],
        context_groups=["short"],
        question_types=["factual"],
        retrieval_k=5,
        iterations_per_question=1,
        checkpoint_interval=2,
        max_questions=8,  # Default limit for quick test
        judge_model=judge_model,  # Use from args or default
        auto_confirm=True,  # Auto-confirm for quick tests to avoid blocking
    )


def create_full_experiment_config(args) -> ExperimentConfig:
    """Create configuration from command line arguments."""
    # Use cost-optimized defaults if --cost-optimized flag is set
    if getattr(args, "cost_optimized", False):
        config = ExperimentConfig.cost_optimized(
            experiment_id=args.experiment_id or f"rerank_value_optimized_{int(time.time())}",
            output_dir=args.output_dir,
            longmemeval_path=args.longmemeval_path,
            document_collections=args.document_collections or ["paul_graham", "arxiv"],
        )
    else:
        # Standard full configuration
        config = ExperimentConfig(
            experiment_id=args.experiment_id or f"rerank_value_{int(time.time())}",
            output_dir=args.output_dir,
            longmemeval_path=args.longmemeval_path,
            document_collections=args.document_collections or ["paul_graham", "arxiv"],
            models=args.models or ["claude-sonnet-3.7"],
            context_groups=args.context_groups or ["short", "medium", "long"],
            question_types=args.question_types or ["factual", "reasoning", "synthesis", "multi_hop"],
            retrieval_k=args.retrieval_k,
            iterations_per_question=args.iterations,
            checkpoint_interval=args.checkpoint_interval,
            judge_model=args.judge_model,
            judge_temperature=args.judge_temperature,
        )

    # Add max_questions if specified
    if getattr(args, "max_questions", None):
        config.max_questions = args.max_questions

    return config


def load_checkpoint_config(checkpoint_file: Path) -> ExperimentConfig:
    """Load configuration from checkpoint file."""
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    with open(checkpoint_file) as f:
        checkpoint_data = json.load(f)

    config_data = checkpoint_data.get("config", {})
    return ExperimentConfig(**config_data)


def resume_experiment(experiment_id: str, results_dir: str = "results/") -> None:
    """Resume an interrupted experiment."""
    results_path = Path(results_dir)
    checkpoint_file = results_path / f"{experiment_id}_checkpoint.json"

    if not checkpoint_file.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint_file}[/red]")
        return

    console.print(f"[blue]Resuming experiment from checkpoint: {checkpoint_file}[/blue]")

    # Load checkpoint
    with open(checkpoint_file) as f:
        checkpoint_data = json.load(f)

    console.print(f"[green]Checkpoint loaded: {checkpoint_data['results_count']} trials completed[/green]")

    # Create new experiment with same config but new ID
    original_config = ExperimentConfig(**checkpoint_data["config"])
    resume_config = ExperimentConfig(
        experiment_id=f"{experiment_id}_resumed_{int(time.time())}",
        output_dir=original_config.output_dir,
        longmemeval_path=original_config.longmemeval_path,
        document_collections=original_config.document_collections,
        models=original_config.models,
        context_groups=original_config.context_groups,
        question_types=original_config.question_types,
        retrieval_k=original_config.retrieval_k,
        iterations_per_question=original_config.iterations_per_question,
        checkpoint_interval=original_config.checkpoint_interval,
        judge_model=original_config.judge_model,
    )

    # Note: In a full implementation, we would restore the exact experiment state
    # For now, we start a new experiment with the same configuration
    console.print("[yellow]Note: Starting new experiment with same configuration[/yellow]")
    console.print("[yellow]Full checkpoint restoration not yet implemented[/yellow]")

    experiment = RerankingValueExperiment(resume_config)
    summary = experiment.run_experiment()

    console.print(f"[green]Resumed experiment completed: {summary.experiment_id}[/green]")


def list_available_experiments(results_dir: str = "results/") -> None:
    """List available experiments and checkpoints."""
    results_path = Path(results_dir)

    if not results_path.exists():
        console.print(f"[yellow]Results directory not found: {results_path}[/yellow]")
        return

    # Find checkpoint files
    checkpoint_files = list(results_path.glob("*_checkpoint.json"))
    result_files = list(results_path.glob("*_results.json"))

    console.print(f"[blue]Available experiments in {results_path}:[/blue]")

    if checkpoint_files:
        console.print("\n[yellow]Checkpoints (can be resumed):[/yellow]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Experiment ID")
        table.add_column("Status")
        table.add_column("Trials Completed")
        table.add_column("Timestamp")

        for checkpoint_file in sorted(checkpoint_files):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)

                experiment_id = data.get("experiment_id", "unknown")
                results_count = data.get("results_count", 0)
                timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(data.get("timestamp", 0)))
                status = "Interrupted" if data.get("interrupted", False) else "In Progress"

                table.add_row(experiment_id, status, str(results_count), timestamp)

            except Exception as e:
                table.add_row(checkpoint_file.stem, "Error", str(e), "Unknown")

        console.print(table)

    if result_files:
        console.print("\n[green]Completed experiments:[/green]")
        table = Table(show_header=True, header_style="bold green")
        table.add_column("Experiment ID")
        table.add_column("Total Trials")
        table.add_column("Best Approach")
        table.add_column("Timestamp")

        for result_file in sorted(result_files):
            try:
                with open(result_file) as f:
                    data = json.load(f)

                experiment_id = data.get("experiment_id", "unknown")
                total_trials = len(data.get("results", []))
                best_approach = data.get("approach_comparison", {}).get("best_accuracy", "Unknown")
                timestamp = time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(data.get("metadata", {}).get("completion_time", 0))
                )

                table.add_row(experiment_id, str(total_trials), best_approach, timestamp)

            except Exception as e:
                table.add_row(result_file.stem, "Error", str(e), "Unknown")

        console.print(table)

    if not checkpoint_files and not result_files:
        console.print("[dim]No experiments found[/dim]")


def validate_config(config: ExperimentConfig) -> bool:
    """Validate experiment configuration with rich display."""
    console.print("🔍 [blue]Validating configuration...[/blue]")

    valid = True
    warnings = []
    errors = []

    # Check required paths
    longmemeval_path = Path(config.longmemeval_path)
    if not longmemeval_path.exists():
        warnings.append(f"LongMemEval path does not exist: {longmemeval_path}")
        warnings.append("Will use sample data from experiment directory")

    # Check models
    if not config.models:
        errors.append("No models specified")
        valid = False

    # Check question types
    valid_question_types = ["factual", "reasoning", "synthesis", "multi_hop"]
    invalid_types = [qt for qt in config.question_types if qt not in valid_question_types]
    if invalid_types:
        errors.append(f"Invalid question types: {invalid_types}")
        errors.append(f"Valid types: {valid_question_types}")
        valid = False

    # Check context groups
    valid_context_groups = ["short", "medium", "long"]
    invalid_groups = [cg for cg in config.context_groups if cg not in valid_context_groups]
    if invalid_groups:
        errors.append(f"Invalid context groups: {invalid_groups}")
        errors.append(f"Valid groups: {valid_context_groups}")
        valid = False

    # Estimate experiment size (note: actual size depends on number of filtered questions)
    # This is a rough estimate - actual size is calculated during experiment setup
    estimated_questions = 10  # Conservative estimate for validation purposes
    total_conditions = (
        estimated_questions
        * len(config.models)
        * 5  # 5 approaches (full_context, basic_rag±rerank, enhanced_rag±rerank)
        * config.iterations_per_question
    )

    # Create validation summary table
    validation_table = Table(title="🔍 Configuration Validation", show_header=True, header_style="bold blue")
    validation_table.add_column("Setting", style="cyan")
    validation_table.add_column("Value", style="yellow")
    validation_table.add_column("Status", justify="center")

    validation_table.add_row("Models", ", ".join(config.models), "✅" if config.models else "❌")
    validation_table.add_row("Context Groups", ", ".join(config.context_groups), "✅" if not invalid_groups else "❌")
    validation_table.add_row("Question Types", ", ".join(config.question_types), "✅" if not invalid_types else "❌")
    validation_table.add_row("Data Path", str(longmemeval_path), "✅" if longmemeval_path.exists() else "⚠️")
    validation_table.add_row(
        "Question Limit",
        str(config.max_questions) if hasattr(config, "max_questions") and config.max_questions else "None",
        "📊" if hasattr(config, "max_questions") and config.max_questions else "✅",
    )
    validation_table.add_row("Estimated Trials", "TBD (depends on questions loaded)", "📊")

    console.print(validation_table)

    # Show warnings if any
    if warnings:
        warning_text = "\n".join([f"• {warning}" for warning in warnings])
        warning_panel = Panel(warning_text, title="⚠️ Validation Warnings", border_style="yellow")
        console.print(warning_panel)

    # Show errors if any
    if errors:
        error_text = "\n".join([f"• {error}" for error in errors])
        error_panel = Panel(error_text, title="❌ Validation Errors", border_style="red")
        console.print(error_panel)

    # Large experiment warning
    if total_conditions > 100:
        large_exp_panel = Panel(
            f"[yellow]This experiment will run {total_conditions} trials, which may take significant time.\n"
            f"Consider using [bold]--quick-test[/bold] for initial validation.[/yellow]",
            title="🚨 Large Experiment Warning",
            border_style="yellow",
        )
        console.print(large_exp_panel)

    return valid


def main() -> None:
    # Load environment variables from root .env file
    root_dir = Path(__file__).parent.parent.parent
    env_file = root_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        console.print(f"[green]✓ Loaded environment from: {env_file}[/green]")
        
        # Ensure OPENAI_API_KEY is set from ORQ_API_KEY for ModelInterface compatibility
        import os
        orq_key = os.getenv('ORQ_API_KEY', '')
        if orq_key:
            os.environ['OPENAI_API_KEY'] = orq_key
            # Print last few chars for verification
            console.print(f"[dim]Using ORQ key ending in: {orq_key[-10:]}[/dim]")
        else:
            console.print("[red]ORQ_API_KEY not found in environment[/red]")
    else:
        console.print(f"[yellow]Warning: .env file not found at {env_file}[/yellow]")
        console.print("[yellow]You may need to set ORQ_API_KEY manually[/yellow]")

    parser = argparse.ArgumentParser(description="Reranking Value Experiment CLI")

    # Action arguments
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with minimal configuration")
    parser.add_argument("--cost-optimized", action="store_true", help="Use cost-optimized configuration for €50 budget")
    parser.add_argument("--resume-from", type=str, help="Resume experiment from checkpoint ID")
    parser.add_argument("--list-experiments", action="store_true", help="List available experiments")

    # Configuration arguments
    parser.add_argument("--experiment-id", type=str, help="Experiment ID (default: auto-generated)")
    parser.add_argument("--output-dir", type=str, default="results/reranking_value/", help="Output directory")
    parser.add_argument("--longmemeval-path", type=str, default="data/LongMemEval/", help="Path to LongMemEval data")
    parser.add_argument("--document-collections", nargs="+", help="Document collections to use")
    parser.add_argument("--models", nargs="+", default=["claude-sonnet-3.7"], help="Models to test")
    parser.add_argument(
        "--context-groups", nargs="+", choices=["short", "medium", "long"], help="Context groups to test"
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        choices=["factual", "reasoning", "synthesis", "multi_hop"],
        help="Question types to test",
    )
    parser.add_argument(
        "--max-questions", type=int, help="Maximum number of questions to process (useful for limiting experiment size)"
    )

    # Experiment parameters
    parser.add_argument("--retrieval-k", type=int, default=20, help="Number of chunks to retrieve")
    parser.add_argument("--iterations", type=int, default=1, help="Iterations per question")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N trials")

    # Judge parameters
    parser.add_argument("--judge-model", type=str, default="gpt-4.1", help="Model to use for evaluation")
    parser.add_argument("--judge-temperature", type=float, default=0.1, help="Temperature for judge model")

    args = parser.parse_args()

    # Handle actions
    if args.list_experiments:
        list_available_experiments(args.output_dir)
        return

    if args.resume_from:
        resume_experiment(args.resume_from, args.output_dir)
        return

    # Create configuration
    if args.quick_test:
        console.print("[blue]Running quick test configuration[/blue]")
        config = create_quick_test_config(args)
        # Override max_questions if specified for quick test
        if args.max_questions:
            config.max_questions = args.max_questions
            console.print(f"[blue]Quick test limited to {args.max_questions} questions[/blue]")
    else:
        config = create_full_experiment_config(args)

    # Validate configuration
    if not validate_config(config):
        console.print("[red]Configuration validation failed[/red]")
        return

    # Display configuration in a nice table
    config_table = Table(title="⚙️ Experiment Configuration", show_header=True, header_style="bold magenta")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="yellow")

    config_table.add_row("Experiment ID", config.experiment_id)
    config_table.add_row("Models", ", ".join(config.models))
    config_table.add_row("Approaches", "Full Context, Basic RAG (±Reranking), Enhanced RAG (±Reranking)")
    config_table.add_row("Context Groups", ", ".join(config.context_groups))
    config_table.add_row("Question Types", ", ".join(config.question_types))
    config_table.add_row("Retrieval K", str(config.retrieval_k))
    config_table.add_row("Iterations per Question", str(config.iterations_per_question))
    config_table.add_row("Judge Model", config.judge_model)
    config_table.add_row("Output Directory", config.output_dir)

    # Show cost optimization status
    is_cost_optimized = (
        len(config.models) == 1
        and config.models[0] == "claude-sonnet-3.7"
        and set(config.context_groups) == {"short", "medium"}
    )
    optimization_status = "✅ Cost-Optimized (~$33)" if is_cost_optimized else "⚠️ Full Configuration"
    config_table.add_row("Budget Status", optimization_status)

    console.print(config_table)

    # The actual trial count will be calculated during experiment initialization
    # based on the number of questions loaded and filtered

    # Run experiment
    console.print(f"\n[bold green]Starting experiment: {config.experiment_id}[/bold green]")

    try:
        experiment = RerankingValueExperiment(config)
        summary = experiment.run_experiment()

        # The experiment itself now handles the completion display nicely
        # Just show final CLI summary with results path
        results_panel = Panel.fit(
            f"[bold green]🎯 Experiment Results Available[/bold green]\n"
            f"[dim]Results Location:[/dim] [blue]{Path(config.output_dir).absolute()}[/blue]\n"
            f"[dim]Analysis Files:[/dim] [cyan]JSON results, statistical analysis, cost breakdown[/cyan]\n"
            f"[dim]Next Steps:[/dim] [yellow]Review results, run additional models, or extend experiment[/yellow]",
            title="📊 Ready for Analysis",
            border_style="green",
        )
        console.print(results_panel)

        # Display quick summary if available
        if summary.approach_comparison:
            approach_stats = summary.approach_comparison.get("approach_statistics", {})
            if approach_stats:
                summary_table = Table(title="🏆 Final Results Summary", show_header=True, header_style="bold green")
                summary_table.add_column("Approach", style="cyan")
                summary_table.add_column("Accuracy", style="yellow", justify="right")
                summary_table.add_column("Performance", style="magenta", justify="center")

                # Sort by accuracy
                sorted_approaches = sorted(approach_stats.items(), key=lambda x: x[1].get("accuracy", 0), reverse=True)

                for i, (approach, stats) in enumerate(sorted_approaches):
                    accuracy = stats.get("accuracy", 0)
                    performance_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
                    summary_table.add_row(approach, f"{accuracy:.1%}", performance_icon)

                console.print(summary_table)

    except KeyboardInterrupt:
        # The experiment itself handles this gracefully now
        pass

    except Exception as e:
        # The experiment itself shows detailed error info
        console.print(f"\n[red]❌ CLI execution failed: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
