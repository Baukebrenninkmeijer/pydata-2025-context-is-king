"""Heatmap plotting functions for reranking analysis."""

import pandas as pd
import plotly.graph_objects as go
import numpy as np


def create_performance_heatmap(df: pd.DataFrame, metric: str = "is_correct") -> go.Figure | None:
    """Create heatmap showing performance across all configurations.

    Args:
        df: DataFrame with experiment results containing columns: k, config_label, and the specified metric
        metric: Performance metric to visualize (default: "is_correct")

    Returns:
        Plotly Figure object or None if metric not available or no valid data

    Raises:
        None (prints warnings for missing data)
    """
    if metric not in df.columns:
        print(f"⚠️  Metric {metric} not available")
        return None

    # Filter valid data
    plot_df = df.dropna(subset=[metric])

    if len(plot_df) == 0:
        print(f"⚠️  No valid data for metric {metric}")
        return None

    # Calculate mean performance by configuration
    # plot_df = plot_df.loc[~((plot_df.retrieval_kind == "full_context") & (plot_df.k != 1))]
    pivot_data = plot_df.groupby(["k", "config_label"])[metric].mean().unstack()
    if pivot_data.empty:
        print(f"⚠️  No data to plot for metric {metric}")
        return None

    # Handle NaN values in the pivot data for text display
    text_values = pivot_data.values.copy()

    # Create text display array, handling NaN values safely
    text_display = np.empty_like(text_values, dtype=object)
    for i in range(text_values.shape[0]):
        for j in range(text_values.shape[1]):
            val = text_values[i, j]
            if pd.isna(val):  # Use pandas isna which handles all types
                text_display[i, j] = ""
            else:
                text_display[i, j] = f"{float(val):.3f}"

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale="RdYlBu_r",
            text=text_display,  # Use the cleaned text values
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="K=%{y}<br>Config=%{x}<br>Score=%{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(title=dict(text=metric.replace("_", " ").title(), font=dict(size=12))),
        )
    )

    fig.update_layout(
        title=f"🔥 Performance Heatmap: {metric.replace('_', ' ').title()}",
        xaxis_title="Configuration",
        yaxis_title="K (Retrieved Documents)",
        height=400,
        width=800,
        # Add log scale to y-axis for better visualization
        yaxis=dict(
            type="log",
            tickmode="array",
            tickvals=sorted(pivot_data.index.dropna()),
            ticktext=[str(int(k)) for k in sorted(pivot_data.index.dropna())],
        ),
        # Improve readability
        xaxis=dict(tickangle=-45),
        # Center the title
        title_x=0.5,
    )

    return fig


def create_config_comparison_heatmap(df: pd.DataFrame, metrics: list[str] | None = None) -> go.Figure | None:
    """Create multi-metric heatmap comparing configurations.

    Args:
        df: DataFrame with experiment results
        metrics: List of metrics to include (default: common performance metrics)

    Returns:
        Plotly Figure object or None if no valid data
    """
    if metrics is None:
        metrics = ["is_correct", "f1_at_1", "mrr", "avg_similarity"]

    # Filter to metrics that exist in the dataframe
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print(f"⚠️  None of the specified metrics {metrics} are available")
        return None

    # Calculate mean performance by configuration for each metric
    config_perf = []

    for metric in available_metrics:
        metric_data = df.dropna(subset=[metric])
        if len(metric_data) > 0:
            means = metric_data.groupby("config_label")[metric].mean()
            for config, value in means.items():
                config_perf.append({"config": config, "metric": metric.replace("_", " ").title(), "value": value})

    if not config_perf:
        print("⚠️  No valid data for any metric")
        return None

    # Create DataFrame and pivot
    perf_df = pd.DataFrame(config_perf)
    pivot_data = perf_df.pivot(index="config", columns="metric", values="value")

    # Handle NaN values for text display
    text_values = pivot_data.values.copy()

    # Create text display array, handling NaN values safely
    text_display = np.empty_like(text_values, dtype=object)
    for i in range(text_values.shape[0]):
        for j in range(text_values.shape[1]):
            val = text_values[i, j]
            if pd.isna(val):  # Use pandas isna which handles all types
                text_display[i, j] = ""
            else:
                text_display[i, j] = f"{float(val):.3f}"

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale="RdYlBu_r",
            text=text_display,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Config=%{y}<br>Metric=%{x}<br>Score=%{z:.3f}<extra></extra>",
            showscale=True,
            colorbar=dict(title=dict(text="Performance Score", font=dict(size=12))),
        )
    )

    fig.update_layout(
        title="🔥 Multi-Metric Configuration Comparison",
        xaxis_title="Metrics",
        yaxis_title="Configuration",
        height=500,
        width=800,
        title_x=0.5,
        xaxis=dict(tickangle=-45),
    )

    return fig
