#!/usr/bin/env python3
"""
Standalone script to run context window scaling experiments.

This script provides a command-line interface to run various experiment configurations.
Run with different options to test specific models, token sizes, and iterations.

Usage:
    python run_experiment.py --help
    python run_experiment.py --quick-test
    python run_experiment.py --full-experiment
    python run_experiment.py --models gpt-5-mini claude-haiku --tokens 1000 10000
"""

import argparse
import sys
from pathlib import Path

# Import from context_is_king module
from context_is_king.experiments.scaling import ContextWindowExperiment, ExperimentAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run context window scaling experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with one model
    python run_experiment.py --quick-test

    # Full experiment (all models, all sizes)
    python run_experiment.py --full-experiment

    # Custom experiment
    python run_experiment.py --models gpt-5-mini claude-haiku --tokens 100 1000 10000 --iterations 5

    # Test specific models only
    python run_experiment.py --models gemini-2.5-flash gemini-2.0-flash-lite --tokens 1000 100000

    # List available checkpoints
    python run_experiment.py --list-checkpoints

    # Resume from checkpoint
    python run_experiment.py --resume-from experiment_abc123

    # Run without saving checkpoints
    python run_experiment.py --quick-test --no-checkpoints

    # Recover and save results from checkpoint (if experiment failed after completion)
    python run_experiment.py --recover-results experiment_abc123
        """,
    )

    # Experiment type (mutually exclusive)
    experiment_group = parser.add_mutually_exclusive_group()
    experiment_group.add_argument(
        "--quick-test", action="store_true", help="Run quick test (gpt-5-mini, small contexts, 2 iterations)"
    )
    experiment_group.add_argument(
        "--full-experiment", action="store_true", help="Run full experiment (all models, all contexts, 3 iterations)"
    )

    # Custom experiment options
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(ContextWindowExperiment.MODELS.keys()),
        help="Models to test (default: all models)",
    )
    parser.add_argument(
        "--tokens", nargs="+", type=int, help="Token sizes to test (default: 10, 100, 1K, 10K, 100K, 1M)"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations per model/size combination (default: 3)"
    )

    # Output options
    parser.add_argument("--output-dir", type=Path, help="Directory to save results (default: data/results)")
    parser.add_argument("--filename", type=str, help="Base filename for results (default: auto-generated timestamp)")
    parser.add_argument("--no-viz", action="store_true", help="Skip generating visualizations")

    # Data directory
    parser.add_argument("--data-dir", type=Path, help="Path to data directory (default: ../../data)")

    # Checkpoint options
    parser.add_argument("--resume-from", type=str, help="Resume experiment from checkpoint ID")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available experiment checkpoints")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable checkpoint saving")
    parser.add_argument("--recover-results", type=str, help="Recover and save results from checkpoint ID")

    return parser.parse_args()


def run_quick_test(experiment: ContextWindowExperiment, resume_from: str = None, save_checkpoints: bool = True):
    """Run a quick test for validation."""
    print("🚀 Running Quick Test...")
    print("- Model: gpt-5-mini")
    print("- Token sizes: 10, 100, 1000")
    print("- Iterations: 2")
    if resume_from:
        print(f"- Resuming from: {resume_from}")
    print("-" * 50)

    return experiment.run_experiment(
        models={"gpt-5-mini": experiment.MODELS["gpt-5-mini"]},
        token_sizes=[10, 100, 1000],
        iterations_per_test=2,
        resume_from=resume_from,
        save_checkpoints=save_checkpoints,
    )


def run_full_experiment(experiment: ContextWindowExperiment, resume_from: str = None, save_checkpoints: bool = True):
    """Run the complete experiment."""
    print("🚀 Running Full Experiment...")
    print(f"- Models: {len(experiment.MODELS)} models")
    print("- Token sizes: 10, 100, 1K, 10K, 100K, 1M")
    print("- Iterations: 3")
    print(f"- Total tests: {len(experiment.MODELS) * len(experiment.DEFAULT_TOKEN_SIZES) * 3}")
    print("- Estimated time: 30-60 minutes")
    if resume_from:
        print(f"- Resuming from: {resume_from}")
    print("-" * 50)

    if not resume_from:
        confirmation = input("This will make ~108 API calls. Continue? [y/N]: ")
        if confirmation.lower() not in ["y", "yes"]:
            print("Experiment cancelled.")
            return None

    return experiment.run_experiment(resume_from=resume_from, save_checkpoints=save_checkpoints)


def run_custom_experiment(
    experiment: ContextWindowExperiment,
    models: list[str],
    tokens: list[int],
    iterations: int,
    resume_from: str = None,
    save_checkpoints: bool = True,
):
    """Run a custom experiment with specified parameters."""

    # Convert model names to model dict
    if models:
        model_dict = {name: experiment.MODELS[name] for name in models}
    else:
        model_dict = experiment.MODELS

    # Use default token sizes if not specified
    if not tokens:
        tokens = [10, 100, 1000, 10000, 100000, 1000000]

    total_tests = len(model_dict) * len(tokens) * iterations

    print("🚀 Running Custom Experiment...")
    print(f"- Models: {list(model_dict.keys())}")
    print(f"- Token sizes: {tokens}")
    print(f"- Iterations: {iterations}")
    print(f"- Total tests: {total_tests}")
    if resume_from:
        print(f"- Resuming from: {resume_from}")
    print("-" * 50)

    if not resume_from and total_tests > 20:
        confirmation = input(f"This will make {total_tests} API calls. Continue? [y/N]: ")
        if confirmation.lower() not in ["y", "yes"]:
            print("Experiment cancelled.")
            return None

    return experiment.run_experiment(
        models=model_dict,
        token_sizes=tokens,
        iterations_per_test=iterations,
        resume_from=resume_from,
        save_checkpoints=save_checkpoints,
    )


def main():
    """Main function to run the experiment."""
    args = parse_args()

    try:
        # Initialize experiment
        print("🔧 Initializing Context Window Experiment...")
        experiment = ContextWindowExperiment(data_dir=args.data_dir)
        print("✅ Connected to ORQ API")
        print(f"📁 Data directory: {experiment.data_dir}")

        # Handle checkpoint listing
        if args.list_checkpoints:
            checkpoints = experiment.list_checkpoints()
            if checkpoints:
                print("\n📋 Available Checkpoints:")
                print("-" * 50)
                for checkpoint in checkpoints:
                    print(f"ID: {checkpoint['experiment_id']}")
                    print(f"Created: {checkpoint['timestamp']}")
                    print(f"Completed: {checkpoint['completed_tasks']}")
                    print(f"Remaining: {checkpoint['remaining_tasks']}")
                    print()
            else:
                print("📋 No checkpoints found.")
            return

        # Handle result recovery
        if args.recover_results:
            print(f"🔄 Recovering results from checkpoint: {args.recover_results}")
            recovered_results = experiment.recover_results_from_checkpoint(args.recover_results)

            if recovered_results:
                # Create analyzer and save results
                analyzer = ExperimentAnalyzer(recovered_results)
                analyzer.print_summary()

                # Save results to experiment directory
                experiment_dir = analyzer.save_results(
                    experiment_id=f"recovered_{args.recover_results}", results_dir=args.output_dir
                )

                print(f"\n✅ Recovered results saved to: {experiment_dir}")
            else:
                print("❌ No results could be recovered.")
            return

        # Check if resuming
        save_checkpoints = not args.no_checkpoints

        # Run the specified experiment
        results = None

        if args.quick_test:
            results = run_quick_test(experiment, args.resume_from, save_checkpoints)
        elif args.full_experiment:
            results = run_full_experiment(experiment, args.resume_from, save_checkpoints)
        else:
            # Custom experiment
            results = run_custom_experiment(
                experiment, args.models, args.tokens, args.iterations, args.resume_from, save_checkpoints
            )

        if results is None:
            print("No experiment was run.")
            return

        # Analyze results
        print("\n" + "=" * 60)
        print("📊 ANALYZING RESULTS")
        print("=" * 60)

        analyzer = ExperimentAnalyzer(results)
        analyzer.print_summary()

        # Save results (includes visualizations unless disabled)
        print("\n💾 Saving results...")
        experiment_dir = analyzer.save_results(experiment_id=args.filename, results_dir=args.output_dir)

        # Clean up checkpoint after successful results saving
        experiment.cleanup_checkpoint()

        print("\n🎉 Experiment complete!")
        print(f"📊 Total results: {len(results)}")
        print(f"✅ Success rate: {sum(1 for r in results if r.success) / len(results):.1%}")

    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error running experiment: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
