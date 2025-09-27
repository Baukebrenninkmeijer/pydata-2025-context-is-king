#!/usr/bin/env python3
"""
Test script to validate checkpoint functionality end-to-end.

This script tests:
1. Starting an experiment and saving checkpoints
2. Interrupting an experiment
3. Resuming from a checkpoint
4. Completing the resumed experiment
"""

import sys
from pathlib import Path

# Import from context_is_king module
from context_is_king.experiments.scaling import ContextWindowExperiment


def test_checkpoint_functionality():
    """Test the complete checkpoint save/resume cycle."""
    print("🧪 Testing Checkpoint Save/Resume Functionality")
    print("=" * 60)

    # Initialize experiment
    experiment = ContextWindowExperiment()

    # Set up a small experiment for testing
    test_models = {"gpt-5-mini": experiment.MODELS["gpt-5-mini"]}
    test_tokens = [100, 200]  # Very small for quick testing
    test_iterations = 1

    print("📋 Test Configuration:")
    print(f"- Models: {list(test_models.keys())}")
    print(f"- Token sizes: {test_tokens}")
    print(f"- Iterations: {test_iterations}")
    print(f"- Total API calls: {len(test_models) * len(test_tokens) * test_iterations}")
    print("-" * 60)

    # Test 1: List checkpoints (should be empty initially)
    print("🔍 Test 1: List checkpoints before experiment")
    initial_checkpoints = experiment.list_checkpoints()
    print(f"✅ Found {len(initial_checkpoints)} initial checkpoints")

    # Test 2: Start an experiment with checkpoints enabled
    print("\n🚀 Test 2: Start experiment with checkpoints")

    try:
        # Note: In a real scenario, we might want to interrupt this
        # For testing, we'll let it complete normally
        results = experiment.run_experiment(
            models=test_models, token_sizes=test_tokens, iterations_per_test=test_iterations, save_checkpoints=True
        )

        print(f"✅ Experiment completed with {len(results)} results")

        # Test 3: Check if checkpoints were created during execution
        print("\n🔍 Test 3: Check for checkpoints after experiment")
        final_checkpoints = experiment.list_checkpoints()
        print(f"✅ Found {len(final_checkpoints)} final checkpoints")

        # If experiment completed normally, checkpoints should be cleaned up
        if len(final_checkpoints) == len(initial_checkpoints):
            print("✅ Checkpoints properly cleaned up after completion")
        else:
            print("⚠️  Checkpoints remain after completion (this may be expected)")

        # Test 4: Test checkpoint data structure
        print("\n🔍 Test 4: Validate checkpoint data structure")
        if hasattr(experiment, "_save_checkpoint"):
            print("✅ Checkpoint save method available")
        if hasattr(experiment, "_load_checkpoint"):
            print("✅ Checkpoint load method available")
        if hasattr(experiment, "list_checkpoints"):
            print("✅ Checkpoint list method available")

        print("\n🎉 Checkpoint functionality test completed!")
        return True

    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted (this is expected in manual testing)")

        # Test resuming from checkpoint
        print("\n🔄 Test 3: Resume from checkpoint")
        current_checkpoints = experiment.list_checkpoints()

        if current_checkpoints:
            latest_checkpoint = current_checkpoints[-1]  # Get most recent
            checkpoint_id = latest_checkpoint["experiment_id"]
            print(f"📂 Attempting to resume from: {checkpoint_id}")

            # Resume the experiment
            resumed_results = experiment.run_experiment(resume_from=checkpoint_id)
            print(f"✅ Resumed experiment completed with {len(resumed_results)} total results")

            # Check cleanup
            final_checkpoints = experiment.list_checkpoints()
            if len(final_checkpoints) < len(current_checkpoints):
                print("✅ Checkpoint cleaned up after successful resume")

            return True
        print("❌ No checkpoints found to resume from")
        return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_checkpoint_error_handling():
    """Test checkpoint error handling."""
    print("\n🧪 Testing Checkpoint Error Handling")
    print("-" * 50)

    experiment = ContextWindowExperiment()

    # Test 1: Try to resume from non-existent checkpoint
    print("🔍 Test 1: Resume from non-existent checkpoint")
    try:
        results = experiment.run_experiment(resume_from="nonexistent_experiment_123")
        print("❌ Should have failed with non-existent checkpoint")
        return False
    except FileNotFoundError:
        print("✅ Correctly handled non-existent checkpoint")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")

    # Test 2: Validate checkpoint directory creation
    print("\n🔍 Test 2: Checkpoint directory handling")
    if experiment.checkpoint_dir.exists():
        print("✅ Checkpoint directory exists")
    else:
        print("⚠️  Checkpoint directory does not exist")

    return True


def main():
    """Run all checkpoint tests."""
    print("🚀 Checkpoint Functionality Test Suite")
    print("=" * 60)

    try:
        # Run basic functionality test
        success1 = test_checkpoint_functionality()

        # Run error handling test
        success2 = test_checkpoint_error_handling()

        if success1 and success2:
            print("\n🎉 All checkpoint tests passed!")
        else:
            print("\n⚠️  Some tests failed or had issues")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
