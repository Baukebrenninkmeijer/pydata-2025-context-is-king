"""
Core plotting utilities and PlotManager class for general functionality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Optional, Tuple, Union, Any, Dict

from .config import (
    PROVIDER_COLORS,
    DEFAULT_FIGURE_SIZES,
    configure_plotting,
    DURATION_FORMATTER,
    THROUGHPUT_FORMATTER,
    TOKEN_FORMATTER
)


class PlotManager:
    """
    General plotting utilities and configuration management.

    This class provides common functionality used across all experiment types,
    including statistical calculations, formatting, and basic plot operations.
    """

    def __init__(self, style: str = 'default', figsize: Optional[Tuple[int, int]] = None):
        """
        Initialize PlotManager with style configuration.

        Args:
            style: Plot style from config (default, publication, presentation)
            figsize: Default figure size override
        """
        self.style = style
        self.figsize = figsize or DEFAULT_FIGURE_SIZES['single']
        configure_plotting(style)

    def configure_style(self, style: str):
        """Configure matplotlib style."""
        self.style = style
        configure_plotting(style)

    def set_figure_size(self, plot_type: str = 'single'):
        """Set figure size based on plot type."""
        if plot_type in DEFAULT_FIGURE_SIZES:
            self.figsize = DEFAULT_FIGURE_SIZES[plot_type]

    @staticmethod
    def calculate_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for the mean using t-distribution.

        Args:
            values: Array of values
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (mean, ci_range)
        """
        n = len(values)
        if n < 2:
            return np.mean(values), 0  # No CI for single points

        mean = np.mean(values)
        sem = stats.sem(values)  # Standard error of mean
        ci_range = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean, ci_range

    def aggregate_by_groups(self, df: pd.DataFrame, group_cols: list, value_col: str,
                           confidence: float = 0.95) -> pd.DataFrame:
        """
        Aggregate DataFrame by groups with confidence intervals.

        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            value_col: Column to calculate statistics for
            confidence: Confidence level

        Returns:
            DataFrame with mean, ci_range, and sample count
        """
        results = []

        for group, group_data in df.groupby(group_cols):
            values = group_data[value_col].values
            mean, ci_range = self.calculate_ci(values, confidence)

            result = dict(zip(group_cols, group if isinstance(group, tuple) else [group]))
            result.update({
                f'{value_col}_mean': mean,
                f'{value_col}_ci_range': ci_range,
                f'{value_col}_count': len(values)
            })
            results.append(result)

        return pd.DataFrame(results)

    def get_provider_color(self, provider_name: str) -> str:
        """Get color for provider with fallback."""
        return PROVIDER_COLORS.get(provider_name, '#666666')

    def create_subplots(self, nrows: int = 1, ncols: int = 1,
                       figsize: Optional[Tuple[int, int]] = None, **kwargs) -> Tuple[plt.Figure, Any]:
        """
        Create subplot figure with consistent styling.

        Args:
            nrows: Number of rows
            ncols: Number of columns
            figsize: Figure size override
            **kwargs: Additional arguments to plt.subplots

        Returns:
            Tuple of (figure, axes)
        """
        figsize = figsize or self.figsize
        return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    def plot_with_ci(self, ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                     yerr: np.ndarray, label: str = None, color: str = None,
                     alpha_fill: float = 0.3, **kwargs) -> None:
        """
        Plot line with confidence interval fill.

        Args:
            ax: Matplotlib axes
            x: X values
            y: Y values (means)
            yerr: Y error values (CI ranges)
            label: Line label
            color: Line color
            alpha_fill: Transparency for CI fill
            **kwargs: Additional arguments to errorbar
        """
        # Plot main line with error bars
        line = ax.errorbar(x, y, yerr=yerr, label=label, color=color,
                          capsize=4, capthick=1.5, **kwargs)

        # Add confidence interval fill
        if color is None:
            color = line[0].get_color()

        ax.fill_between(x, y - yerr, y + yerr, alpha=alpha_fill, color=color)

    def apply_log_scale(self, ax: plt.Axes, x: bool = True, y: bool = True):
        """Apply log scaling to axes."""
        if x:
            ax.set_xscale('log')
        if y:
            ax.set_yscale('log')

    def apply_formatters(self, ax: plt.Axes, x_type: str = None, y_type: str = None):
        """
        Apply appropriate formatters to axes.

        Args:
            ax: Matplotlib axes
            x_type: Type for x-axis (duration, throughput, tokens)
            y_type: Type for y-axis (duration, throughput, tokens)
        """
        formatter_map = {
            'duration': DURATION_FORMATTER,
            'throughput': THROUGHPUT_FORMATTER,
            'tokens': TOKEN_FORMATTER
        }

        if x_type and x_type in formatter_map:
            ax.xaxis.set_major_formatter(formatter_map[x_type])
        if y_type and y_type in formatter_map:
            ax.yaxis.set_major_formatter(formatter_map[y_type])

    def save_figure(self, fig: plt.Figure, filename: str,
                   output_dir: Optional[Union[str, Path]] = None,
                   dpi: int = 150, **kwargs) -> Path:
        """
        Save figure with consistent settings.

        Args:
            fig: Figure to save
            filename: Filename (with or without extension)
            output_dir: Directory to save in (default: IMG_DIR from context_is_king)
            dpi: DPI for saved figure
            **kwargs: Additional arguments to savefig

        Returns:
            Path to saved file
        """
        # Get default output directory
        if output_dir is None:
            try:
                from context_is_king import IMG_DIR
                output_dir = IMG_DIR
            except ImportError:
                output_dir = Path.cwd() / 'img'

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Add .png extension if needed
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            filename += '.png'

        filepath = output_dir / filename
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', **kwargs)
        return filepath

    def line_plot_with_groups(self, df: pd.DataFrame, x_col: str, y_col: str,
                             group_col: str, confidence: float = 0.95,
                             log_x: bool = False, log_y: bool = False,
                             **kwargs) -> plt.Figure:
        """
        Create line plot with groups and confidence intervals.

        Args:
            df: DataFrame with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            group_col: Column to group by
            confidence: Confidence level for CI
            log_x: Use log scale for x-axis
            log_y: Use log scale for y-axis
            **kwargs: Additional arguments

        Returns:
            matplotlib Figure
        """
        # Aggregate data by groups
        agg_df = self.aggregate_by_groups(df, [group_col, x_col], y_col, confidence)

        fig, ax = self.create_subplots()

        for group in df[group_col].unique():
            group_data = agg_df[agg_df[group_col] == group].sort_values(x_col)
            color = self.get_provider_color(group)

            self.plot_with_ci(
                ax,
                group_data[x_col],
                group_data[f'{y_col}_mean'],
                group_data[f'{y_col}_ci_range'],
                label=group,
                color=color,
                marker='o',
                linewidth=2,
                markersize=6
            )

        if log_x or log_y:
            self.apply_log_scale(ax, log_x, log_y)

        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def comparison_plot(self, df: pd.DataFrame, metric: str, groupby: str,
                       plot_type: str = 'box', **kwargs) -> plt.Figure:
        """
        Create comparison plot (box plot, violin plot, etc.).

        Args:
            df: DataFrame with data
            metric: Column to plot
            groupby: Column to group by
            plot_type: Type of plot (box, violin, strip)
            **kwargs: Additional arguments

        Returns:
            matplotlib Figure
        """
        fig, ax = self.create_subplots()

        if plot_type == 'box':
            bp = ax.boxplot([df[df[groupby] == group][metric].values
                           for group in df[groupby].unique()],
                          labels=df[groupby].unique(),
                          patch_artist=True)

            # Color boxes by provider
            for patch, group in zip(bp['boxes'], df[groupby].unique()):
                patch.set_facecolor(self.get_provider_color(group))
                patch.set_alpha(0.7)

        ax.set_xlabel(groupby.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig