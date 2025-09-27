"""
Context Window Scaling Experiment Package

This package provides tools to measure and analyze how LLM call duration
scales with context window size across different models and providers.

Main Components:
- ContextWindowExperiment: Main experiment runner
- ExperimentAnalyzer: Results analysis and visualization
- ExperimentResult: Data structure for individual test results

Quick Start:
    from context_scaling import ContextWindowExperiment, ExperimentAnalyzer

    experiment = ContextWindowExperiment()
    results = experiment.run_experiment()
    analyzer = ExperimentAnalyzer(results)
    analyzer.print_summary()
    analyzer.create_visualizations()
"""

from context_is_king.experiments.scaling import (
    ContextWindowExperiment,
    ExperimentAnalyzer,
    ExperimentResult,
    quick_test,
    full_experiment,
)

__version__ = "1.0.0"
__author__ = "PyData 2025 Context is King Project"

__all__ = ["ContextWindowExperiment", "ExperimentAnalyzer", "ExperimentResult", "quick_test", "full_experiment"]
