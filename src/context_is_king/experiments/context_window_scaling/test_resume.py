#!/usr/bin/env python3
"""
Test script to manually test the checkpoint resume functionality.

This script starts an experiment, shows the checkpoint, then demonstrates resuming.
"""

import sys
from pathlib import Path

# Import from context_is_king module
from context_is_king.experiments.scaling import ContextWindowExperiment


def create_interrupted_experiment():
    """Create an experiment that can be interrupted and resumed."""
    print("🚀 Creating Interruptible Experiment")
    print("=" * 50)

    experiment = ContextWindowExperiment()

    # Set up an experiment with multiple API calls
    test_models = {"gpt-5-mini": experiment.MODELS["gpt-5-mini"]}
    test_tokens = [100, 200, 300]  # 3 API calls
    test_iterations = 1

    print(f"📋 Experiment will make {len(test_tokens)} API calls")
    print(f"🔄 After the first API call completes, press Ctrl+C to interrupt")
    print(f"📂 We'll then demonstrate resuming from the checkpoint")
    print("-" * 50)

    try:
        results = experiment.run_experiment(
            models=test_models, token_sizes=test_tokens, iterations_per_test=test_iterations, save_checkpoints=True
        )

        print(f"\n✅ Experiment completed normally with {len(results)} results")
        print("🔄 Since it wasn't interrupted, let's still demonstrate list/resume functionality")

    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted!")

        # Show available checkpoints
        checkpoints = experiment.list_checkpoints()
        if checkpoints:
            print(f"\n📋 Found {len(checkpoints)} checkpoint(s):")
            latest = checkpoints[-1]
            print(f"   ID: {latest['experiment_id']}")
            print(f"   Completed: {latest['completed_tasks']}")
            print(f"   Remaining: {latest['remaining_tasks']}")

            # Ask user if they want to resume
            response = input(f"\n🔄 Resume from checkpoint {latest['experiment_id']}? [y/N]: ")
            if response.lower() in ["y", "yes"]:
                print(f"\n📂 Resuming experiment...")
                resumed_results = experiment.run_experiment(resume_from=latest["experiment_id"])
                print(f"✅ Resumed experiment completed with {len(resumed_results)} total results")
            else:
                print("⏸️  Leaving experiment in checkpoint state")
        else:
            print("❌ No checkpoints found (experiment may have been too short)")

    return experiment


def main():
    """Main test function."""
    print("🧪 Manual Checkpoint Resume Test")
    print("=" * 60)

    experiment = create_interrupted_experiment()

    # Show final checkpoint status
    final_checkpoints = experiment.list_checkpoints()
    print(f"\n📋 Final checkpoint count: {len(final_checkpoints)}")

    if final_checkpoints:
        print("🔄 Available for resume:")
        for cp in final_checkpoints:
            print(f"   - {cp['experiment_id']}: {cp['completed_tasks']} completed, {cp['remaining_tasks']} remaining")
    else:
        print("🗑️  No checkpoints remain (cleaned up after completion)")

    print("\n✅ Manual test completed!")


if __name__ == "__main__":
    main()
