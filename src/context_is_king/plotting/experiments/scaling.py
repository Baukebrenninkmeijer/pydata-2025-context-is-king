"""
Scaling experiment-specific plotting functionality.

Contains ScalingPlotter class with specific methods for each visualization type
used in context window scaling experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Union, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

from ..core import PlotManager
from ..config import PROVIDER_COLORS, DEFAULT_FIGURE_SIZES


class ScalingPlotter:
    """
    Context window scaling experiment visualizations.

    Provides specific methods for each type of scaling analysis plot
    with consistent styling and comprehensive batch generation.
    """

    def __init__(self, df: pd.DataFrame, plot_manager: Optional[PlotManager] = None):
        """
        Initialize ScalingPlotter with experiment data.

        Args:
            df: DataFrame with experiment results
            plot_manager: PlotManager instance (creates default if None)
        """
        self.df = self._prepare_dataframe(df)
        self.pm = plot_manager or PlotManager()

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame with provider extraction and filtering.

        Args:
            df: Raw experiment DataFrame

        Returns:
            Cleaned DataFrame with provider column
        """
        df = df[df["success"]].copy()  # Filter successful results only

        # Extract provider from model names
        def get_provider(model_name):
            if "gpt" in model_name.lower():
                return "OpenAI"
            elif "gemini" in model_name.lower():
                return "Google"
            elif "claude" in model_name.lower():
                return "Anthropic"
            else:
                return "Unknown"

        df["provider"] = df["model_name"].apply(get_provider)

        # Remove unknown providers
        df = df[df["provider"] != "Unknown"]

        return df

    def plot_duration_by_provider(self, show_ci: bool = True, log_y: bool = True,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Provider-grouped duration vs context size with confidence intervals.

        Args:
            show_ci: Show 95% confidence intervals
            log_y: Use log scale for y-axis
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = self.pm.create_subplots(figsize=DEFAULT_FIGURE_SIZES['single'])

        # Aggregate data by provider and context size
        agg_df = self.pm.aggregate_by_groups(
            self.df, ['provider', 'context_size'], 'duration_seconds'
        )

        for provider in self.df['provider'].unique():
            provider_data = agg_df[agg_df['provider'] == provider].sort_values('context_size')
            color = self.pm.get_provider_color(provider)

            if show_ci:
                self.pm.plot_with_ci(
                    ax,
                    provider_data['context_size'],
                    provider_data['duration_seconds_mean'],
                    provider_data['duration_seconds_ci_range'],
                    label=provider,
                    color=color,
                    marker='o',
                    linewidth=2,
                    markersize=6
                )
            else:
                ax.plot(
                    provider_data['context_size'],
                    provider_data['duration_seconds_mean'],
                    label=provider,
                    color=color,
                    marker='o',
                    linewidth=2,
                    markersize=6
                )

            # Add individual data points with transparency
            individual_data = self.df[self.df['provider'] == provider]
            ax.scatter(
                individual_data['context_size'],
                individual_data['duration_seconds'],
                alpha=0.3, s=15, color=color
            )

        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        ax.set_xlabel('Context Size (tokens)')
        ax.set_ylabel('Duration (seconds)')
        title = 'Context Size vs Response Duration by Provider'
        if show_ci:
            title += '\n(Mean ± 95% Confidence Interval)'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.pm.apply_formatters(ax, y_type='duration')
        plt.tight_layout()

        if save_path:
            self.pm.save_figure(fig, save_path)

        return fig

    def plot_throughput_by_provider(self, show_ci: bool = True, log_y: bool = True,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Provider-grouped throughput vs context size.

        Args:
            show_ci: Show 95% confidence intervals
            log_y: Use log scale for y-axis
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = self.pm.create_subplots(figsize=DEFAULT_FIGURE_SIZES['single'])

        # Aggregate data by provider and context size
        agg_df = self.pm.aggregate_by_groups(
            self.df, ['provider', 'context_size'], 'tokens_per_second'
        )

        for provider in self.df['provider'].unique():
            provider_data = agg_df[agg_df['provider'] == provider].sort_values('context_size')
            color = self.pm.get_provider_color(provider)

            if show_ci:
                self.pm.plot_with_ci(
                    ax,
                    provider_data['context_size'],
                    provider_data['tokens_per_second_mean'],
                    provider_data['tokens_per_second_ci_range'],
                    label=provider,
                    color=color,
                    marker='o',
                    linewidth=2,
                    markersize=6
                )
            else:
                ax.plot(
                    provider_data['context_size'],
                    provider_data['tokens_per_second_mean'],
                    label=provider,
                    color=color,
                    marker='o',
                    linewidth=2,
                    markersize=6
                )

            # Add individual data points
            individual_data = self.df[self.df['provider'] == provider]
            ax.scatter(
                individual_data['context_size'],
                individual_data['tokens_per_second'],
                alpha=0.3, s=15, color=color
            )

        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        ax.set_xlabel('Context Size (tokens)')
        ax.set_ylabel('Throughput (tokens/second)')
        title = 'Context Size vs Processing Throughput by Provider'
        if show_ci:
            title += '\n(Mean ± 95% Confidence Interval)'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.pm.apply_formatters(ax, y_type='throughput')
        plt.tight_layout()

        if save_path:
            self.pm.save_figure(fig, save_path)

        return fig

    def plot_duration_throughput_dual(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Side-by-side duration and throughput comparison.

        Args:
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = self.pm.create_subplots(1, 2, figsize=DEFAULT_FIGURE_SIZES['dual'])

        # Aggregate data by provider and context size
        dur_agg = self.pm.aggregate_by_groups(
            self.df, ['provider', 'context_size'], 'duration_seconds'
        )
        thr_agg = self.pm.aggregate_by_groups(
            self.df, ['provider', 'context_size'], 'tokens_per_second'
        )

        for provider in self.df['provider'].unique():
            color = self.pm.get_provider_color(provider)

            # Duration plot
            dur_data = dur_agg[dur_agg['provider'] == provider].sort_values('context_size')
            self.pm.plot_with_ci(
                ax1,
                dur_data['context_size'],
                dur_data['duration_seconds_mean'],
                dur_data['duration_seconds_ci_range'],
                label=provider,
                color=color,
                marker='o',
                linewidth=2,
                markersize=6
            )

            # Throughput plot
            thr_data = thr_agg[thr_agg['provider'] == provider].sort_values('context_size')
            self.pm.plot_with_ci(
                ax2,
                thr_data['context_size'],
                thr_data['tokens_per_second_mean'],
                thr_data['tokens_per_second_ci_range'],
                label=provider,
                color=color,
                marker='o',
                linewidth=2,
                markersize=6
            )

        # Configure duration plot
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Context Size (tokens)')
        ax1.set_ylabel('Duration (seconds)')
        ax1.set_title('Response Duration vs Context Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        self.pm.apply_formatters(ax1, y_type='duration')

        # Configure throughput plot
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Context Size (tokens)')
        ax2.set_ylabel('Throughput (tokens/second)')
        ax2.set_title('Processing Throughput vs Context Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        self.pm.apply_formatters(ax2, y_type='throughput')

        plt.tight_layout()

        if save_path:
            self.pm.save_figure(fig, save_path)

        return fig

    def plot_polynomial_fits(self, degree: int = 2, show_r2: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Polynomial fitting in log-log space with R² scores.

        Args:
            degree: Polynomial degree (default 2 for quadratic)
            show_r2: Show R² scores in labels
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = self.pm.create_subplots(1, 2, figsize=DEFAULT_FIGURE_SIZES['dual'])

        polynomial_fits = {}

        for provider in self.df['provider'].unique():
            provider_data = self.df[self.df['provider'] == provider].copy()
            color = self.pm.get_provider_color(provider)

            # Get context sizes and durations
            X = provider_data['context_size'].values
            y = provider_data['duration_seconds'].values

            # Transform to log space for fitting
            log_X = np.log10(X).reshape(-1, 1)
            log_y = np.log10(y)

            # Fit polynomial: log(y) = a*log(x)^2 + b*log(x) + c
            poly_features = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly_features.fit_transform(log_X)

            poly_reg = LinearRegression()
            poly_reg.fit(X_poly, log_y)

            # Calculate R² score
            y_pred = poly_reg.predict(X_poly)
            r2 = r2_score(log_y, y_pred)

            # Store fit results
            polynomial_fits[provider] = {
                'coefficients': poly_reg.coef_,
                'intercept': poly_reg.intercept_,
                'r2': r2
            }

            # Create smooth curve for plotting
            x_smooth = np.logspace(np.log10(X.min()), np.log10(X.max()), 100)
            log_x_smooth = np.log10(x_smooth).reshape(-1, 1)
            X_smooth_poly = poly_features.transform(log_x_smooth)
            log_y_smooth = poly_reg.predict(X_smooth_poly)
            y_smooth = 10**log_y_smooth

            # Labels with R² if requested
            data_label = f'{provider} (data)'
            if show_r2:
                fit_label = f'{provider} fit (R²={r2:.3f})'
            else:
                fit_label = f'{provider} fit'

            # Plot 1: Log-log scale
            ax1.scatter(X, y, alpha=0.4, s=20, color=color, label=data_label)
            ax1.loglog(x_smooth, y_smooth, '--', linewidth=3, color=color, label=fit_label)

            # Plot 2: Linear scale
            ax2.scatter(X, y, alpha=0.4, s=20, color=color, label=data_label)
            ax2.semilogx(x_smooth, y_smooth, '--', linewidth=3, color=color, label=fit_label)

        # Configure plots
        for ax, title_suffix in [(ax1, '(Log-Log Scale)'), (ax2, '(Linear Scale)')]:
            ax.set_xlabel('Context Size (tokens)')
            ax.set_ylabel('Duration (seconds)')
            ax.set_title(f'Provider Scaling with Polynomial Fits\n{title_suffix}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.pm.apply_formatters(ax, y_type='duration')

        plt.tight_layout()

        if save_path:
            self.pm.save_figure(fig, save_path)

        # Store fits for later analysis
        self._polynomial_fits = polynomial_fits

        return fig

    def plot_theoretical_comparison(self, include_exponential: bool = True,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Provider data vs linear/quadratic/exponential theoretical scaling.

        Args:
            include_exponential: Include exponential scaling reference
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = self.pm.create_subplots(1, 2, figsize=DEFAULT_FIGURE_SIZES['dual'])

        # Aggregate provider data
        agg_df = self.pm.aggregate_by_groups(
            self.df, ['provider', 'context_size'], 'duration_seconds'
        )

        # Plot provider data
        for provider in self.df['provider'].unique():
            provider_data = agg_df[agg_df['provider'] == provider].sort_values('context_size')
            color = self.pm.get_provider_color(provider)

            for ax in [ax1, ax2]:
                self.pm.plot_with_ci(
                    ax,
                    provider_data['context_size'],
                    provider_data['duration_seconds_mean'],
                    provider_data['duration_seconds_ci_range'],
                    label=provider,
                    color=color,
                    marker='o',
                    linewidth=2,
                    markersize=6
                )

        # Add theoretical scaling lines
        context_range = np.logspace(0, 6, 100)  # 1 to 1M tokens
        base_duration = 1.0

        # Linear scaling
        linear_scaling = base_duration * (context_range / 1)
        for ax in [ax1, ax2]:
            ax.plot(context_range, linear_scaling, '--', color='gray', alpha=0.7,
                   linewidth=2, label='Linear (2x)')

        # Quadratic scaling
        quadratic_scaling = base_duration * (context_range / 1000)**2
        for ax in [ax1, ax2]:
            ax.plot(context_range, quadratic_scaling, ':', color='black', alpha=0.7,
                   linewidth=2, label='Quadratic (x²)')

        # Exponential scaling
        if include_exponential:
            exponential_scaling = base_duration * (2 ** (context_range / 1000))
            for ax in [ax1, ax2]:
                ax.plot(context_range, exponential_scaling, linestyle='dashdot',
                       color='dimgrey', alpha=0.7, linewidth=2, label='Exponential (2ˣ)')

        # Configure axes
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Context Size (tokens)')
        ax1.set_ylabel('Duration (seconds)')
        ax1.set_title('Provider Scaling vs Theoretical References\n(Log-Log Scale)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xscale('log')
        ax2.set_xlabel('Context Size (tokens)')
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_title('Provider Scaling vs Theoretical References\n(Linear Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        for ax in [ax1, ax2]:
            self.pm.apply_formatters(ax, y_type='duration')

        plt.tight_layout()

        if save_path:
            self.pm.save_figure(fig, save_path)

        return fig

    def plot_scaling_examples(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Educational scaling relationship examples (linear, quadratic, exponential).

        Args:
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = self.pm.create_subplots(1, 2, figsize=DEFAULT_FIGURE_SIZES['dual'])

        # Create example data
        x_values = np.array([1, 10, 100, 1000, 10000, 100_000])
        base_time = 1

        # Define scaling functions
        y_linear = base_time * 2 * x_values
        y_quadratic = base_time * (x_values ** 2) / 1000
        y_exponential = base_time * (2 ** (x_values / 1000))

        # Plot on both scales
        for ax, title_suffix in [(ax1, '(Log-Log Scale)'), (ax2, '(Linear Scale - Shows Explosion!)')]:
            ax.plot(x_values, y_linear, '--', linewidth=2, label='Linear: y = 2x',
                   color='gray', alpha=0.8)
            ax.plot(x_values, y_quadratic, ':', linewidth=2, label='Quadratic: y = x²',
                   color='black', alpha=0.8)
            ax.plot(x_values, y_exponential, linestyle='dashdot', linewidth=2,
                   label='Exponential: y = 2ˣ', color='dimgrey', alpha=0.8)

            ax.set_xlabel('Context Size (tokens)')
            ax.set_ylabel('Response Time (seconds)')
            ax.set_title(f'Scaling Relationships Comparison\n{title_suffix}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')

        plt.tight_layout()

        if save_path:
            self.pm.save_figure(fig, save_path)

        return fig

    def plot_provider_summary_dashboard(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        4-panel dashboard: duration, throughput, polynomial, theoretical.

        Args:
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=DEFAULT_FIGURE_SIZES['dashboard'])

        # Create 2x2 grid
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4)

        # Aggregate data
        agg_df = self.pm.aggregate_by_groups(
            self.df, ['provider', 'context_size'], 'duration_seconds'
        )

        for provider in self.df['provider'].unique():
            provider_data = agg_df[agg_df['provider'] == provider].sort_values('context_size')
            color = self.pm.get_provider_color(provider)

            # Panel 1: Duration
            self.pm.plot_with_ci(
                ax1, provider_data['context_size'], provider_data['duration_seconds_mean'],
                provider_data['duration_seconds_ci_range'], label=provider, color=color,
                marker='o', linewidth=2, markersize=4
            )

            # Panel 2: Throughput
            thr_agg = self.pm.aggregate_by_groups(
                self.df, ['provider', 'context_size'], 'tokens_per_second'
            )
            thr_data = thr_agg[thr_agg['provider'] == provider].sort_values('context_size')
            self.pm.plot_with_ci(
                ax2, thr_data['context_size'], thr_data['tokens_per_second_mean'],
                thr_data['tokens_per_second_ci_range'], label=provider, color=color,
                marker='o', linewidth=2, markersize=4
            )

            # Panel 3 & 4: Theoretical comparison
            for ax in [ax3, ax4]:
                self.pm.plot_with_ci(
                    ax, provider_data['context_size'], provider_data['duration_seconds_mean'],
                    provider_data['duration_seconds_ci_range'], label=provider, color=color,
                    marker='o', linewidth=2, markersize=4
                )

        # Add theoretical lines to panels 3 & 4
        context_range = np.logspace(0, 6, 100)
        base_duration = 1.0
        linear_scaling = base_duration * (context_range / 1)
        quadratic_scaling = base_duration * (context_range / 1000)**2

        for ax in [ax3, ax4]:
            ax.plot(context_range, linear_scaling, '--', color='gray', alpha=0.7,
                   linewidth=1, label='Linear')
            ax.plot(context_range, quadratic_scaling, ':', color='black', alpha=0.7,
                   linewidth=1, label='Quadratic')

        # Configure panels
        panels = [
            (ax1, 'Duration (Log-Log)', 'duration'),
            (ax2, 'Throughput (Log-Log)', 'throughput'),
            (ax3, 'vs Theoretical (Log-Log)', 'duration'),
            (ax4, 'vs Theoretical (Linear)', 'duration')
        ]

        for ax, title, format_type in panels:
            ax.set_xscale('log')
            if 'Linear' not in title:
                ax.set_yscale('log')

            ax.set_xlabel('Context Size')
            ax.set_ylabel(format_type.title())
            ax.set_title(title)
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)
            self.pm.apply_formatters(ax, y_type=format_type)

        plt.tight_layout()

        if save_path:
            self.pm.save_figure(fig, save_path)

        return fig

    def generate_all_plots(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, plt.Figure]:
        """
        Generate all standard scaling plots at once.

        Args:
            output_dir: Directory to save plots (optional)

        Returns:
            Dictionary of plot names to matplotlib figures
        """
        plots = {}

        # Generate each plot type
        plots['duration_by_provider'] = self.plot_duration_by_provider(
            save_path='duration_by_provider.png' if output_dir else None
        )

        plots['throughput_by_provider'] = self.plot_throughput_by_provider(
            save_path='throughput_by_provider.png' if output_dir else None
        )

        plots['duration_throughput_dual'] = self.plot_duration_throughput_dual(
            save_path='duration_throughput_dual.png' if output_dir else None
        )

        plots['polynomial_fits'] = self.plot_polynomial_fits(
            save_path='polynomial_fits.png' if output_dir else None
        )

        plots['theoretical_comparison'] = self.plot_theoretical_comparison(
            save_path='theoretical_comparison.png' if output_dir else None
        )

        plots['scaling_examples'] = self.plot_scaling_examples(
            save_path='scaling_examples.png' if output_dir else None
        )

        plots['summary_dashboard'] = self.plot_provider_summary_dashboard(
            save_path='summary_dashboard.png' if output_dir else None
        )

        return plots

    def calculate_scaling_exponents(self) -> pd.DataFrame:
        """
        Calculate power-law exponents for each provider.

        Returns:
            DataFrame with scaling statistics by provider
        """
        results = []

        for provider in self.df['provider'].unique():
            provider_data = self.df[self.df['provider'] == provider]

            if len(provider_data) >= 3:
                # Calculate slope in log-log space
                log_size = np.log10(provider_data['context_size'])
                log_duration = np.log10(provider_data['duration_seconds'])

                slope, intercept, r_value, p_value, std_err = stats.linregress(log_size, log_duration)

                results.append({
                    'provider': provider,
                    'scaling_exponent': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'std_error': std_err,
                    'n_samples': len(provider_data)
                })

        return pd.DataFrame(results)

    def get_scaling_statistics(self) -> pd.DataFrame:
        """
        Get comprehensive scaling statistics table.

        Returns:
            DataFrame with detailed scaling analysis
        """
        stats_df = self.calculate_scaling_exponents()

        # Add interpretation
        def interpret_scaling(exponent):
            if exponent < 0.8:
                return "Sub-linear (efficient)"
            elif exponent < 1.2:
                return "Near-linear"
            elif exponent < 2.0:
                return "Super-linear"
            else:
                return "Quadratic or worse"

        stats_df['scaling_interpretation'] = stats_df['scaling_exponent'].apply(interpret_scaling)

        return stats_df