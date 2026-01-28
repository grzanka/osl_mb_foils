"""Dose distribution plotting utilities."""

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def create_dose_colormap(white_threshold_percent: float = 1.0) -> LinearSegmentedColormap:
    """Create a custom colormap for dose visualization.
    
    The colormap transitions: white -> green (50%) -> red (100%)
    with a white region for low doses.
    
    Parameters
    ----------
    white_threshold_percent : float
        Percentage of vmax below which colors appear white. Default 1.0%.
        
    Returns
    -------
    LinearSegmentedColormap
        Custom colormap for dose visualization
    """
    threshold_frac = white_threshold_percent / 100.0
    cmap_dict = {
        'red':   [(0.0, 1.0, 1.0),
                  (threshold_frac, 1.0, 1.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 1.0, 1.0)],
        'green': [(0.0, 1.0, 1.0),
                  (threshold_frac, 1.0, 1.0),
                  (0.5, 0.5, 0.5),
                  (1.0, 0.0, 0.0)],
        'blue':  [(0.0, 1.0, 1.0),
                  (threshold_frac, 1.0, 1.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0)]
    }
    return LinearSegmentedColormap('custom_dose', cmap_dict)


def plot_2d_dose(
    dose_array: npt.NDArray,
    px_to_mm: float,
    title: str = 'Dose Distribution',
    vmin: float = 0,
    vmax: Optional[float] = None,
    white_threshold_percent: float = 1.0,
    figsize: tuple = (10, 8),
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes, plt.cm.ScalarMappable]:
    """Plot 2D dose distribution with proper axes in mm.
    
    Parameters
    ----------
    dose_array : np.ndarray
        2D dose array
    px_to_mm : float
        Pixel to mm conversion factor
    title : str
        Plot title
    vmin : float
        Minimum value for color scale. Default 0.
    vmax : float, optional
        Maximum value for color scale. If None, uses data max.
    white_threshold_percent : float
        Percentage of vmax below which colors appear white. Default 1.0%.
    figsize : tuple
        Figure size (width, height) in inches. Default (10, 8).
    ax : Axes, optional
        Existing matplotlib axes to plot on. If None, creates new figure.
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes
    im : ScalarMappable
        Image object (for colorbar manipulation)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Create extent for proper axis scaling [left, right, bottom, top]
    height, width = dose_array.shape
    extent = [0, width * px_to_mm, height * px_to_mm, 0]
    
    # Determine vmax if not provided
    if vmax is None:
        vmax = dose_array.max()
    
    cmap = create_dose_colormap(white_threshold_percent)
    
    im = ax.imshow(dose_array, cmap=cmap, vmin=vmin, vmax=vmax, 
                   extent=extent, aspect='equal')
    
    ax.set_xlabel('X [mm]', fontsize=12)
    ax.set_ylabel('Y [mm]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    fig.colorbar(im, ax=ax, label='Dose [Gy]', fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig, ax, im


def plot_profiles(
    dose_array: npt.NDArray,
    px_to_mm: float,
    x_position: Optional[int] = None,
    y_position: Optional[int] = None,
    profile_width: int = 5,
    title: str = 'Dose Profiles',
    figsize: tuple = (12, 5)
) -> Tuple[Figure, Tuple[Axes, Axes], Tuple[npt.NDArray, npt.NDArray]]:
    """Plot horizontal and vertical dose profiles.
    
    Parameters
    ----------
    dose_array : np.ndarray
        2D dose array
    px_to_mm : float
        Pixel to mm conversion factor
    x_position : int, optional
        Row index for horizontal profile (default: center)
    y_position : int, optional
        Column index for vertical profile (default: center)
    profile_width : int
        Width of averaging region in pixels. Default 5.
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height) in inches. Default (12, 5).
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : tuple
        Tuple of (ax_horizontal, ax_vertical)
    profiles : tuple
        Tuple of (horizontal_profile, vertical_profile) arrays
    """
    height, width = dose_array.shape
    
    # Default positions at center
    if x_position is None:
        x_position = height // 2
    if y_position is None:
        y_position = width // 2
    
    # Calculate profiles with averaging
    hw = profile_width // 2
    
    # Horizontal profile (along x-axis at given row)
    h_profile = dose_array[max(0, x_position-hw):min(height, x_position+hw), :].mean(axis=0)
    h_x_mm = np.arange(len(h_profile)) * px_to_mm
    
    # Vertical profile (along y-axis at given column)
    v_profile = dose_array[:, max(0, y_position-hw):min(width, y_position+hw)].mean(axis=1)
    v_y_mm = np.arange(len(v_profile)) * px_to_mm
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    ax1.plot(h_x_mm, h_profile, 'b-', linewidth=2)
    ax1.set_xlabel('X Position [mm]', fontsize=11)
    ax1.set_ylabel('Dose [Gy]', fontsize=11)
    ax1.set_title(f'Horizontal Profile at Y={x_position*px_to_mm:.1f} mm', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, None)
    
    ax2.plot(v_y_mm, v_profile, 'r-', linewidth=2)
    ax2.set_xlabel('Y Position [mm]', fontsize=11)
    ax2.set_ylabel('Dose [Gy]', fontsize=11)
    ax2.set_title(f'Vertical Profile at X={y_position*px_to_mm:.1f} mm', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, None)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig, (ax1, ax2), (h_profile, v_profile)


def plot_dose_with_profiles(
    dose_array: npt.NDArray,
    px_to_mm: float,
    title: str = '',
    vmax: Optional[float] = None,
    white_threshold_percent: float = 1.0,
    profile_width: int = 10,
    figsize: tuple = (16, 5)
) -> Figure:
    """Create a combined plot with 2D dose and horizontal/vertical profiles.
    
    Parameters
    ----------
    dose_array : np.ndarray
        2D dose array
    px_to_mm : float
        Pixel to mm conversion factor
    title : str
        Overall figure title
    vmax : float, optional
        Maximum value for color scale. If None, uses 99th percentile.
    white_threshold_percent : float
        Percentage of vmax below which colors appear white. Default 1.0%.
    profile_width : int
        Width of averaging region for profiles in pixels. Default 10.
    figsize : tuple
        Figure size (width, height) in inches. Default (16, 5).
    
    Returns
    -------
    fig : Figure
        Matplotlib figure with 3 subplots
    """
    height, width = dose_array.shape
    
    # Calculate vmax from 99th percentile of positive values
    if vmax is None:
        positive_values = dose_array[dose_array > 0]
        vmax = np.percentile(positive_values, 99) if len(positive_values) > 0 else dose_array.max()
    
    cmap = create_dose_colormap(white_threshold_percent)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=figsize)
    
    # 2D dose plot
    ax1 = plt.subplot(1, 3, 1)
    extent = [0, width * px_to_mm, height * px_to_mm, 0]
    im = ax1.imshow(dose_array, cmap=cmap, vmin=0, vmax=vmax, 
                    extent=extent, aspect='equal')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_title('2D Dose Distribution')
    plt.colorbar(im, ax=ax1, label='Dose [Gy]', fraction=0.046)
    
    # Profile positions at center
    y_pos = height // 2
    x_pos = width // 2
    y_pos_mm = y_pos * px_to_mm
    x_pos_mm = x_pos * px_to_mm
    
    # Add profile position lines on 2D dose plot
    ax1.axhline(y=y_pos_mm, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='Horizontal profile')
    ax1.axvline(x=x_pos_mm, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Vertical profile')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Half-width for averaging
    hw = profile_width // 2
    
    # Horizontal profile
    ax2 = plt.subplot(1, 3, 2)
    h_profile = dose_array[max(0, y_pos-hw):min(height, y_pos+hw), :].mean(axis=0)
    h_x_mm = np.arange(len(h_profile)) * px_to_mm
    ax2.plot(h_x_mm, h_profile, 'b-', linewidth=2)
    ax2.set_xlabel('X Position [mm]')
    ax2.set_ylabel('Dose [Gy]')
    ax2.set_title(f'Horizontal Profile (Y={y_pos_mm:.1f} mm)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, None)
    
    # Vertical profile
    ax3 = plt.subplot(1, 3, 3)
    v_profile = dose_array[:, max(0, x_pos-hw):min(width, x_pos+hw)].mean(axis=1)
    v_y_mm = np.arange(len(v_profile)) * px_to_mm
    ax3.plot(v_y_mm, v_profile, 'r-', linewidth=2)
    ax3.set_xlabel('Y Position [mm]')
    ax3.set_ylabel('Dose [Gy]')
    ax3.set_title(f'Vertical Profile (X={x_pos_mm:.1f} mm)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, None)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_profile_comparison(
    profile_data: List[Tuple[npt.NDArray, float, str]],
    orientation: str = 'horizontal',
    figsize: tuple = (10, 6)
) -> Figure:
    """Plot multiple profiles on the same axes for comparison.
    
    Parameters
    ----------
    profile_data : list of tuples
        Each tuple contains (dose_cropped, px_to_mm, label)
    orientation : str
        'horizontal' or 'vertical' profile orientation. Default 'horizontal'.
    figsize : tuple
        Figure size (width, height) in inches. Default (10, 6).
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(profile_data)))
    
    for idx, (dose_cropped, px_to_mm, label) in enumerate(profile_data):
        height, width = dose_cropped.shape
        hw = 5  # half-width for averaging
        
        if orientation == 'horizontal':
            y_pos = height // 2
            profile = dose_cropped[max(0, y_pos-hw):min(height, y_pos+hw), :].mean(axis=0)
            x_mm = np.arange(len(profile)) * px_to_mm
            ax.set_xlabel('X Position [mm]', fontsize=12)
            ax.set_title('Horizontal Profiles Comparison', fontsize=14, fontweight='bold')
        else:
            x_pos = width // 2
            profile = dose_cropped[:, max(0, x_pos-hw):min(width, x_pos+hw)].mean(axis=1)
            x_mm = np.arange(len(profile)) * px_to_mm
            ax.set_xlabel('Y Position [mm]', fontsize=12)
            ax.set_title('Vertical Profiles Comparison', fontsize=14, fontweight='bold')
        
        ax.plot(x_mm, profile, '-', linewidth=2, color=colors[idx], label=label)
    
    ax.set_ylabel('Dose [Gy]', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    return fig
