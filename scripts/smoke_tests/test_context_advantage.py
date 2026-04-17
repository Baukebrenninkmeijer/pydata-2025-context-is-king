#!/usr/bin/env python3
"""
Quick test experiment to verify the context advantage system works.
This runs a minimal experiment with just 2 models and smaller parameters.
"""

import subprocess
import sys
from pathlib import Path


def run_test_experiment():
    """Run a minimal test experiment."""

    # Get the path to the main experiment script
    script_path = Path(__file__).parent / "context_advantage_experiment.py"

    # Test experiment parameters - much smaller scale
    cmd = [
        sys.executable,
        str(script_path),
        "--experiment-type",
        "both",  # Test both needle and longmemeval
        "--models",
        "gemini-2.5-flash-lite",
        "--context-sizes",
        "10000,50000",  # Just 2 context sizes
        "--iterations",
        "1",  # Just 1 iteration
        "--needles-per-haystack",
        "3",  # Just 3 needles
    ]

    print("🧪 Running test experiment with minimal parameters:")  # noqa: T201
    print("   Models: gemini-2.5-flash-lite")
    print("   Context sizes: 10k, 50k")
    print("   Iterations: 1")
    print("   Needles: 3 per haystack")
    print("   Expected total: ~20 experiments")
    print()

    # Run the experiment
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Test experiment completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test experiment failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Test experiment interrupted by user")
        return False

    return True


if __name__ == "__main__":
    success = run_test_experiment()
    sys.exit(0 if success else 1)
