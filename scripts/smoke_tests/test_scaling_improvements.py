#!/usr/bin/env python3
"""
Test script to validate the context window scaling improvements.

This script tests:
1. Token limit filtering for gpt-5-nano
2. 429 error parsing functionality
3. Model information display
"""

import sys
from pathlib import Path

# Import from context_is_king module
from context_is_king.experiments.scaling import ContextWindowExperiment


def test_token_filtering():
    """Test that token sizes are properly filtered based on model limits."""
    print("🧪 Testing Token Size Filtering")
    print("-" * 40)

    experiment = ContextWindowExperiment()

    # Test with gpt-5-nano (272k limit)
    large_token_sizes = [1000, 10000, 100000, 500000, 1000000]
    filtered = experiment._filter_token_sizes_for_model("gpt-5-nano", large_token_sizes)

    print(f"Original sizes: {large_token_sizes}")
    print(f"Filtered sizes: {filtered}")
    print(f"Expected: [1000, 10000, 100000] (500k and 1M should be filtered out)")

    # Test with gpt-5-mini (no filtering expected)
    filtered_mini = experiment._filter_token_sizes_for_model("gpt-5-mini", large_token_sizes)
    print(f"\nGPT-5-Mini filtered: {filtered_mini}")
    print(f"Expected: {large_token_sizes} (no filtering)")

    # Test with claude-haiku (200k limit)
    filtered_haiku = experiment._filter_token_sizes_for_model("claude-haiku", large_token_sizes)
    print(f"\nClaude Haiku filtered: {filtered_haiku}")
    print(f"Expected: [1000, 10000, 100000] (500k and 1M should be filtered out)")

    # Test with unknown model
    filtered_unknown = experiment._filter_token_sizes_for_model("unknown-model", large_token_sizes)
    print(f"\nUnknown model filtered: {filtered_unknown}")
    print(f"Expected: {large_token_sizes} (no filtering)")

    print("✅ Token filtering test complete\n")


def test_429_parsing():
    """Test 429 error message parsing."""
    print("🧪 Testing 429 Error Parsing")
    print("-" * 40)

    experiment = ContextWindowExperiment()

    # Test error message from the example
    error_msg = "Error code: 429 - {'code': 429, 'error': 'Requests to the ChatCompletions_Create Operation under Azure OpenAI API version 2025-01-01-preview have exceeded token rate limit of your current OpenAI S0 pricing tier. Please retry after 44 seconds. Please go here: https://aka.ms/oai/quotaincrease if you would like to further increase the default rate limit. For Free Account customers, upgrade to Pay as you Go here: https://aka.ms/429TrialUpgrade.', 'source': 'provider'}"

    delay = experiment._extract_retry_delay(error_msg)
    print(f"Error message: {error_msg[:100]}...")
    print(f"Extracted delay: {delay} seconds")
    print(f"Expected: 44 seconds")

    # Test other patterns
    test_cases = [
        ("Please retry after 30 seconds", 30),
        ("try again in 15 seconds", 15),
        ("Rate limit exceeded, retry after 60 seconds", 60),
        ("No delay mentioned", 60),  # Should default to 60
    ]

    for msg, expected in test_cases:
        delay = experiment._extract_retry_delay(msg)
        status = "✅" if delay == expected else "❌"
        print(f"{status} '{msg}' -> {delay}s (expected: {expected}s)")

    print("✅ 429 parsing test complete\n")


def test_model_display():
    """Test model information display."""
    print("🧪 Testing Model Information Display")
    print("-" * 40)

    experiment = ContextWindowExperiment()
    experiment.display_model_info()

    print("\n✅ Model display test complete\n")


def main():
    """Run all tests."""
    print("🚀 Testing Context Window Scaling Improvements")
    print("=" * 60)

    try:
        test_token_filtering()
        test_429_parsing()
        test_model_display()

        print("🎉 All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
