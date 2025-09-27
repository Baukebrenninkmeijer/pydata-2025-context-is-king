#!/usr/bin/env python3
"""
Demo script to create a checkpoint that you can then list and resume from.
This simulates an interrupted experiment.
"""

import sys
import json
from pathlib import Path

# Import from context_is_king module
from context_is_king.experiments.scaling import ContextWindowExperiment, ExperimentCheckpoint, ExperimentResult
import pandas as pd


def create_demo_checkpoint():
    """Create a demo checkpoint file manually."""
    print("📁 Creating Demo Checkpoint for Testing...")

    experiment = ContextWindowExperiment()

    # Create some fake completed results
    completed_results = [
        ExperimentResult(
            model_name="gpt-5-mini",
            context_size=100,
            duration_seconds=0.85,
            tokens_per_second=117.6,
            success=True,
            timestamp=pd.Timestamp.now().isoformat(),
            iteration=1,
        ),
        ExperimentResult(
            model_name="gpt-5-mini",
            context_size=200,
            duration_seconds=1.12,
            tokens_per_second=178.5,
            success=True,
            timestamp=pd.Timestamp.now().isoformat(),
            iteration=1,
        ),
    ]

    # Create some remaining tasks
    remaining_tasks = [
        ("gpt-5-mini", "azure/gpt-5-mini", 500, 1),
        ("claude-haiku", "anthropic/claude-3-haiku-20240307", 100, 1),
        ("claude-haiku", "anthropic/claude-3-haiku-20240307", 200, 1),
        ("claude-haiku", "anthropic/claude-3-haiku-20240307", 500, 1),
    ]

    # Create experiment config
    config = {
        "models": {"gpt-5-mini": "azure/gpt-5-mini", "claude-haiku": "anthropic/claude-3-haiku-20240307"},
        "token_sizes": [100, 200, 500],
        "iterations_per_test": 1,
    }

    # Generate demo experiment ID
    experiment_id = "demo_checkpoint_test_12345"

    # Save the checkpoint manually using the experiment's method
    experiment._save_checkpoint(experiment_id, completed_results, remaining_tasks, config)

    print(f"✅ Demo checkpoint created: {experiment_id}")
    print(f"📊 Completed: {len(completed_results)} tasks")
    print(f"📋 Remaining: {len(remaining_tasks)} tasks")

    return experiment_id


def main():
    """Create demo checkpoint and show listing."""
    checkpoint_id = create_demo_checkpoint()

    print(f"\n🔍 Now testing checkpoint listing...")
    experiment = ContextWindowExperiment()
    checkpoints = experiment.list_checkpoints()

    if checkpoints:
        print("\n📋 Available Checkpoints:")
        print("-" * 50)
        for i, checkpoint in enumerate(checkpoints):
            print(f"{i + 1}. ID: {checkpoint['experiment_id']}")
            print(f"   Created: {checkpoint['timestamp']}")
            print(f"   Completed: {checkpoint['completed_tasks']} tasks")
            print(f"   Remaining: {checkpoint['remaining_tasks']} tasks")
            print()
    else:
        print("❌ No checkpoints found")

    print(f"\n💡 You can now test resuming with:")
    print(f"   python run_experiment.py --resume-from {checkpoint_id}")
    print(f"\n💡 Or list checkpoints with:")
    print(f"   python run_experiment.py --list-checkpoints")


if __name__ == "__main__":
    main()
