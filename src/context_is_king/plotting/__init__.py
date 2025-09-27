"""
Context Is King Plotting Module

High-level plotting API for experiment visualizations with specific functions
for each plot type and convenient batch generation.
"""

from .core import PlotManager
from .config import PROVIDER_COLORS, configure_plotting
from .experiments.scaling import ScalingPlotter
from . import reranking

# High-level convenience functions for quick plotting
def plot_scaling_duration(df, **kwargs):
    """
    Quick function to plot duration scaling by provider.

    Args:
        df: DataFrame with experiment results
        **kwargs: Arguments passed to plot_duration_by_provider()

    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    plotter = ScalingPlotter(df)
    return plotter.plot_duration_by_provider(**kwargs)


def plot_scaling_throughput(df, **kwargs):
    """
    Quick function to plot throughput scaling by provider.

    Args:
        df: DataFrame with experiment results
        **kwargs: Arguments passed to plot_throughput_by_provider()

    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    plotter = ScalingPlotter(df)
    return plotter.plot_throughput_by_provider(**kwargs)


def plot_scaling_polynomial(df, **kwargs):
    """
    Quick function to plot polynomial fits in log-log space.

    Args:
        df: DataFrame with experiment results
        **kwargs: Arguments passed to plot_polynomial_fits()

    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    plotter = ScalingPlotter(df)
    return plotter.plot_polynomial_fits(**kwargs)


def plot_scaling_theoretical(df, **kwargs):
    """
    Quick function to plot theoretical scaling comparison.

    Args:
        df: DataFrame with experiment results
        **kwargs: Arguments passed to plot_theoretical_comparison()

    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    plotter = ScalingPlotter(df)
    return plotter.plot_theoretical_comparison(**kwargs)


def create_scaling_report(df, output_dir=None):
    """
    Generate complete scaling analysis report with all plots.

    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots (optional)

    Returns:
        dict: Dictionary of plot names to matplotlib figures
    """
    plotter = ScalingPlotter(df)
    return plotter.generate_all_plots(output_dir)


__all__ = [
    'PlotManager',
    'ScalingPlotter',
    'PROVIDER_COLORS',
    'configure_plotting',
    'plot_scaling_duration',
    'plot_scaling_throughput',
    'plot_scaling_polynomial',
    'plot_scaling_theoretical',
    'create_scaling_report',
    'reranking'
]