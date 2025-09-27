"""
Plotting configuration including colors, styles, and formatters.
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# Provider color mapping (consistent across all plots)
PROVIDER_COLORS = {
    "OpenAI": "#10A37F",  # OpenAI green
    "Google": "#4285F4",  # Google blue
    "Anthropic": "#FF6B35",  # Anthropic orange
}

# Plot styles configuration
PLOT_STYLES = {
    'default': {
        'figure.figsize': (10, 8),
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'lines.markersize': 6,
    },
    'publication': {
        'figure.figsize': (12, 8),
        'font.size': 14,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
    },
    'presentation': {
        'figure.figsize': (16, 10),
        'font.size': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 3,
        'lines.markersize': 10,
    }
}

# Default figure sizes for different plot types
DEFAULT_FIGURE_SIZES = {
    'single': (10, 8),
    'dual': (16, 6),
    'dashboard': (20, 12),
    'comparison': (14, 8),
}


def duration_formatter(x, pos):
    """
    Format duration values to avoid scientific notation.

    Args:
        x: Value to format
        pos: Position (unused)

    Returns:
        str: Formatted string
    """
    if x >= 1:
        return f'{x:.1f}'      # "2.5" instead of "2.5e+00"
    elif x >= 0.1:
        return f'{x:.2f}'      # "0.25" instead of "2.5e-01"
    else:
        return f'{x:.3f}'      # "0.025" instead of "2.5e-02"


def throughput_formatter(x, pos):
    """
    Format throughput values to avoid scientific notation.

    Args:
        x: Value to format
        pos: Position (unused)

    Returns:
        str: Formatted string
    """
    if x >= 10000:
        return f'{x:,.0f}'     # "50,000" instead of "5e+04"
    elif x >= 1000:
        return f'{x:.0f}'      # "5000" instead of "5e+03"
    elif x >= 100:
        return f'{x:.0f}'      # "500" instead of "5e+02"
    else:
        return f'{x:.1f}'      # "50.5" instead of "5.05e+01"


def token_formatter(x, pos):
    """
    Format token counts with K/M notation.

    Args:
        x: Value to format
        pos: Position (unused)

    Returns:
        str: Formatted string
    """
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x/1_000:.0f}K'
    else:
        return f'{x:.0f}'


def percentage_formatter(x, pos):
    """
    Format percentage values.

    Args:
        x: Value to format (0-1 range)
        pos: Position (unused)

    Returns:
        str: Formatted string
    """
    return f'{x*100:.1f}%'


def configure_plotting(style='default', dpi=150):
    """
    Configure matplotlib with consistent styling.

    Args:
        style: Style name from PLOT_STYLES
        dpi: DPI for figures
    """
    if style in PLOT_STYLES:
        plt.rcParams.update(PLOT_STYLES[style])

    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['savefig.bbox'] = 'tight'


def get_provider_color(provider_name):
    """
    Get color for a provider, with fallback.

    Args:
        provider_name: Name of the provider

    Returns:
        str: Hex color code
    """
    return PROVIDER_COLORS.get(provider_name, '#666666')  # Gray fallback


# Create FuncFormatter objects for easy use
DURATION_FORMATTER = FuncFormatter(duration_formatter)
THROUGHPUT_FORMATTER = FuncFormatter(throughput_formatter)
TOKEN_FORMATTER = FuncFormatter(token_formatter)
PERCENTAGE_FORMATTER = FuncFormatter(percentage_formatter)