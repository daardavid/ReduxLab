# heatmap_visualization.py
"""
Heatmap Visualization Module for Correlation Analysis

Provides professional heatmap visualization using Seaborn with customizable
color palettes, annotations, and export capabilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.patches as mpatches
from pathlib import Path

# Optional imports for interactive visualization
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import logging
try:
    from logging_config import get_logger
    logger = get_logger("heatmap_visualization")
except ImportError:
    import logging
    logger = logging.getLogger("heatmap_visualization")


class HeatmapVisualizer:
    """Professional heatmap visualization for correlation matrices."""

    def __init__(self):
        self.logger = logger

        # Default configuration
        self.default_config = {
            'figsize': (12, 10),
            'cmap': 'coolwarm',
            'annot': False,
            'fmt': '.2f',
            'annot_size': 8,
            'title': 'Correlation Heatmap',
            'title_size': 16,
            'label_size': 12,
            'tick_size': 10,
            'cbar_label': 'Correlation',
            'cbar_shrink': 0.8,
            'mask_diagonal': True,
            'square': True,
            'linewidths': 0.5,
            'linecolor': 'white',
            'auto_rotation': True,  # New: automatically determine best label rotation
            'force_horizontal': False,  # New: force horizontal labels (0 degrees)
            'max_label_length': 20  # New: maximum characters to show in labels
        }

    def _calculate_optimal_rotation_and_size(self, labels: List[str], matrix_size: int) -> Dict[str, Any]:
        """
        Calculate optimal rotation angle and font size for axis labels.
        
        Args:
            labels: List of label names
            matrix_size: Size of correlation matrix
            
        Returns:
            Dict with 'rotation', 'fontsize', and 'truncated_labels'
        """
        # Calculate average and maximum label length
        avg_length = np.mean([len(str(label)) for label in labels])
        max_length = max([len(str(label)) for label in labels])
        
        # Truncate very long labels
        max_label_length = self.default_config.get('max_label_length', 20)
        truncated_labels = []
        for label in labels:
            label_str = str(label)
            if len(label_str) > max_label_length:
                truncated_labels.append(label_str[:max_label_length-3] + '...')
            else:
                truncated_labels.append(label_str)
        
        # Determine optimal settings based on matrix size and label characteristics
        if matrix_size <= 10:
            # Small matrices: use horizontal labels with larger font
            rotation = 0
            fontsize = max(8, min(12, 100 // matrix_size))
        elif matrix_size <= 20:
            # Medium matrices: use horizontal if labels are short, otherwise 45°
            if avg_length <= 8:
                rotation = 0
                fontsize = max(7, min(10, 80 // matrix_size))
            else:
                rotation = 45
                fontsize = max(7, min(9, 80 // matrix_size))
        elif matrix_size <= 40:
            # Large matrices: prefer 45° rotation with smaller font
            rotation = 45
            fontsize = max(6, min(8, 60 // matrix_size))
        else:
            # Very large matrices: use 90° rotation with very small font
            rotation = 90
            fontsize = max(5, min(7, 50 // matrix_size))
        
        self.logger.info(f"Optimal label settings: rotation={rotation}°, fontsize={fontsize}, "
                        f"matrix_size={matrix_size}, avg_length={avg_length:.1f}")
        
        return {
            'rotation': rotation,
            'fontsize': fontsize,
            'truncated_labels': truncated_labels
        }

    def create_heatmap(self, correlation_matrix: pd.DataFrame,
                      config: Optional[Dict[str, Any]] = None,
                      ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Create a professional correlation heatmap.

        Args:
            correlation_matrix: Correlation/similarity matrix
            config: Visualization configuration
            ax: Optional matplotlib axes

        Returns:
            Matplotlib figure object
        """
        try:
            self.logger.info("Creating correlation heatmap")

            if correlation_matrix.empty:
                raise ValueError("Correlation matrix is empty")

            # Merge configurations
            viz_config = self.default_config.copy()
            if config:
                viz_config.update(config)

            # Create figure if not provided
            if ax is None:
                # Use larger figure size to accommodate many labels
                # Adjust figure size based on matrix size
                matrix_size = len(correlation_matrix)
                if matrix_size <= 15:
                    fig, ax = plt.subplots(figsize=(12, 10))
                elif matrix_size <= 30:
                    fig, ax = plt.subplots(figsize=(16, 14))
                else:
                    fig, ax = plt.subplots(figsize=(20, 18))
            else:
                fig = ax.get_figure()

            # Prepare data
            data = correlation_matrix.copy()
            
            # Calculate optimal label settings
            labels = list(data.columns)
            matrix_size = len(data)
            
            # For this correction: X-axis always diagonal, Y-axis always horizontal
            # The force_horizontal option is kept for backwards compatibility but doesn't change the behavior
            label_settings = self._calculate_optimal_rotation_and_size(labels, matrix_size)

            # Mask diagonal if requested
            if viz_config.get('mask_diagonal', True):
                mask = np.eye(len(data), dtype=bool)
            else:
                mask = None

            # Create heatmap
            heatmap = sns.heatmap(
                data,
                ax=ax,
                cmap=viz_config['cmap'],
                annot=viz_config['annot'],
                fmt=viz_config['fmt'],
                annot_kws={'size': viz_config['annot_size']},
                mask=mask,
                square=viz_config['square'],
                linewidths=viz_config['linewidths'],
                linecolor=viz_config['linecolor'],
                cbar_kws={
                    'shrink': viz_config['cbar_shrink'],
                    'label': viz_config['cbar_label']
                }
            )

            # Apply optimal label settings
            # Update X-axis labels - SIEMPRE EN DIAGONAL (45°) para evitar sobreposición
            ax.set_xticklabels(
                label_settings['truncated_labels'],
                rotation=45,  # FIJO - siempre 45° diagonal para el eje X
                ha='right',   # Alineación derecha para labels diagonales
                fontsize=label_settings['fontsize']
            )
            
            # Update Y-axis labels - HORIZONTALES (0°) para mejor legibilidad
            ax.set_yticklabels(
                label_settings['truncated_labels'],
                rotation=0,   # FIJO - siempre horizontal para el eje Y
                ha='right',   # Alineación derecha
                fontsize=label_settings['fontsize']
            )

            # Customize appearance
            ax.set_title(viz_config['title'], fontsize=viz_config['title_size'], pad=20)

            # Set axis labels
            ax.set_xlabel('Research Units', fontsize=viz_config['label_size'])
            ax.set_ylabel('Research Units', fontsize=viz_config['label_size'])

            # Adjust layout with extra space for rotated labels
            if label_settings['rotation'] > 0:
                plt.tight_layout()
                # Add extra bottom margin for angled labels
                plt.subplots_adjust(bottom=0.15)
            else:
                plt.tight_layout()

            self.logger.info(f"Heatmap created successfully with {label_settings['rotation']}° label rotation")
            return fig

        except Exception as e:
            error_msg = f"Error creating heatmap: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def create_annotated_heatmap(self, correlation_matrix: pd.DataFrame,
                                threshold: float = 0.5,
                                config: Optional[Dict[str, Any]] = None) -> plt.Figure:
        """
        Create heatmap with selective annotations based on threshold.

        Args:
            correlation_matrix: Correlation matrix
            threshold: Minimum absolute correlation for annotation
            config: Visualization configuration

        Returns:
            Matplotlib figure
        """
        try:
            # Create annotation mask
            annot_mask = correlation_matrix.abs() >= threshold

            # Update config for annotations
            viz_config = self.default_config.copy()
            if config:
                viz_config.update(config)

            viz_config.update({
                'annot': annot_mask,
                'fmt': '.2f'
            })

            return self.create_heatmap(correlation_matrix, viz_config)

        except Exception as e:
            self.logger.error(f"Error creating annotated heatmap: {e}")
            raise

    def create_clustered_heatmap(self, correlation_matrix: pd.DataFrame,
                               config: Optional[Dict[str, Any]] = None) -> plt.Figure:
        """
        Create heatmap with hierarchical clustering of rows and columns.

        Args:
            correlation_matrix: Correlation matrix
            config: Visualization configuration

        Returns:
            Matplotlib figure with clustered heatmap
        """
        try:
            self.logger.info("Creating clustered heatmap")

            if correlation_matrix.empty:
                raise ValueError("Correlation matrix is empty")

            # Merge configurations
            viz_config = self.default_config.copy()
            if config:
                viz_config.update(config)

            # Create figure
            fig, ax = plt.subplots(figsize=viz_config['figsize'])

            # Create clustered heatmap using seaborn
            cg = sns.clustermap(
                correlation_matrix,
                cmap=viz_config['cmap'],
                annot=viz_config['annot'],
                fmt=viz_config['fmt'],
                annot_kws={'size': viz_config['annot_size']},
                figsize=viz_config['figsize'],
                cbar_pos=(0.02, 0.8, 0.05, 0.18),
                tree_kws={'linewidths': 1.5}
            )

            # Customize
            cg.ax_heatmap.set_title(viz_config['title'], fontsize=viz_config['title_size'], pad=20)

            # Adjust colorbar
            cg.ax_cbar.set_ylabel(viz_config['cbar_label'], fontsize=viz_config['label_size'])

            plt.tight_layout()

            self.logger.info("Clustered heatmap created successfully")
            return cg.fig

        except Exception as e:
            error_msg = f"Error creating clustered heatmap: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def create_similarity_histogram(self, correlation_matrix: pd.DataFrame,
                                 config: Optional[Dict[str, Any]] = None) -> plt.Figure:
        """
        Create histogram of similarity values for threshold selection guidance.

        Args:
            correlation_matrix: Correlation matrix
            config: Visualization configuration

        Returns:
            Matplotlib figure with histogram
        """
        try:
            self.logger.info("Creating similarity histogram")

            # Extract off-diagonal values
            off_diagonal = correlation_matrix.values
            np.fill_diagonal(off_diagonal, np.nan)
            similarities = off_diagonal[~np.isnan(off_diagonal)].flatten()

            if len(similarities) == 0:
                raise ValueError("No similarity values to plot")

            # Default histogram config
            hist_config = {
                'figsize': (10, 6),
                'bins': 30,
                'alpha': 0.7,
                'color': 'skyblue',
                'edgecolor': 'black',
                'title': 'Distribution of Similarity Values',
                'xlabel': 'Similarity',
                'ylabel': 'Frequency'
            }

            if config:
                hist_config.update(config)

            # Create figure
            fig, ax = plt.subplots(figsize=hist_config['figsize'])

            # Plot histogram
            n, bins, patches = ax.hist(
                similarities,
                bins=hist_config['bins'],
                alpha=hist_config['alpha'],
                color=hist_config['color'],
                edgecolor=hist_config['edgecolor']
            )

            # Add vertical line at mean
            mean_sim = np.mean(similarities)
            ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_sim:.3f}')

            # Add vertical line at median
            median_sim = np.median(similarities)
            ax.axvline(median_sim, color='green', linestyle='--', linewidth=2,
                      label=f'Median: {median_sim:.3f}')

            # Customize
            ax.set_title(hist_config['title'], fontsize=14, pad=20)
            ax.set_xlabel(hist_config['xlabel'], fontsize=12)
            ax.set_ylabel(hist_config['ylabel'], fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            self.logger.info("Similarity histogram created successfully")
            return fig

        except Exception as e:
            error_msg = f"Error creating similarity histogram: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def create_interactive_heatmap(self, correlation_matrix: pd.DataFrame,
                                  title: str = 'Interactive Correlation Heatmap',
                                  config: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Create an interactive heatmap using Plotly for better scalability with many units.

        Args:
            correlation_matrix: Correlation/similarity matrix
            title: Title for the heatmap
            config: Optional configuration dict

        Returns:
            Plotly figure object
        """
        try:
            if not PLOTLY_AVAILABLE:
                raise ImportError("Plotly is required for interactive heatmap. Install with: pip install plotly")

            self.logger.info("Creating interactive heatmap")

            if correlation_matrix.empty:
                raise ValueError("Correlation matrix is empty")

            # Merge configurations
            viz_config = self.default_config.copy()
            if config:
                viz_config.update(config)

            # Calculate optimal label settings for font size only
            labels = list(correlation_matrix.columns)
            matrix_size = len(correlation_matrix)
            
            # Fixed rotation: X=45°, Y=0°, but dynamic font size
            label_settings = self._calculate_optimal_rotation_and_size(labels, matrix_size)
            fontsize = label_settings['fontsize']
            x_labels = label_settings['truncated_labels']
            y_labels = label_settings['truncated_labels']

            # Create Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=x_labels,
                y=y_labels,
                colorscale='RdBu_r',  # Red-Blue reversed, good for correlations
                zmid=0,  # Center the color scale at 0
                hoverongaps=False,
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))

            # Calculate appropriate figure size based on matrix size
            if matrix_size <= 15:
                width, height = 700, 600
            elif matrix_size <= 30:
                width, height = 900, 800
            else:
                width, height = 1100, 1000

            # Update layout for better appearance
            fig.update_layout(
                title={
                    'text': title,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=16)
                },
                xaxis=dict(
                    showgrid=False,
                    tickangle=45,  # SIEMPRE 45° diagonal para eje X (evita sobreposición)
                    tickfont=dict(size=fontsize),
                    side='bottom'
                ),
                yaxis=dict(
                    showgrid=False,
                    autorange='reversed',  # Puts diagonal from top-left to bottom-right
                    tickangle=0,   # SIEMPRE horizontal (0°) para eje Y
                    tickfont=dict(size=fontsize)
                ),
                width=width,
                height=height,
                margin=dict(
                    l=max(100, matrix_size * 3),  # Dynamic left margin based on matrix size
                    r=100, 
                    t=100, 
                    b=max(100, 80)  # Fixed bottom margin for diagonal X labels
                )
            )

            # Add colorbar configuration
            fig.update_traces(
                colorbar=dict(
                    title="Correlation",
                    tickformat=".2f"
                )
            )

            self.logger.info(f"Interactive heatmap created successfully with fixed rotation (X=45°, Y=0°)")
            return fig

        except Exception as e:
            error_msg = f"Error creating interactive heatmap: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def export_heatmap(self, fig: plt.Figure, filepath: str,
                      dpi: int = 300, format: str = 'png') -> bool:
        """
        Export heatmap to file.

        Args:
            fig: Matplotlib figure
            filepath: Output file path
            dpi: Resolution for raster formats
            format: Export format ('png', 'svg', 'pdf', etc.)

        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Exporting heatmap to {filepath}")

            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Save figure
            fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight')

            self.logger.info(f"Heatmap exported successfully to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting heatmap: {e}")
            return False

    def get_available_colormaps(self) -> List[str]:
        """Get list of available matplotlib colormaps."""
        return [
            'coolwarm', 'RdYlBu', 'RdYlGn', 'viridis', 'plasma', 'inferno', 'magma',
            'Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'Greys',
            'bwr', 'seismic', 'Spectral', 'RdBu'
        ]


# Convenience functions
def create_correlation_heatmap(correlation_matrix: pd.DataFrame,
                             config: Optional[Dict[str, Any]] = None) -> plt.Figure:
    """Convenience function for creating correlation heatmaps."""
    visualizer = HeatmapVisualizer()
    return visualizer.create_heatmap(correlation_matrix, config)


def create_annotated_heatmap(correlation_matrix: pd.DataFrame,
                           threshold: float = 0.5,
                           config: Optional[Dict[str, Any]] = None) -> plt.Figure:
    """Convenience function for annotated heatmaps."""
    visualizer = HeatmapVisualizer()
    return visualizer.create_annotated_heatmap(correlation_matrix, threshold, config)


def create_clustered_heatmap(correlation_matrix: pd.DataFrame,
                           config: Optional[Dict[str, Any]] = None) -> plt.Figure:
    """Convenience function for clustered heatmaps."""
    visualizer = HeatmapVisualizer()
    return visualizer.create_clustered_heatmap(correlation_matrix, config)


def create_similarity_histogram(correlation_matrix: pd.DataFrame,
                             config: Optional[Dict[str, Any]] = None) -> plt.Figure:
    """Convenience function for similarity histograms."""
    visualizer = HeatmapVisualizer()
    return visualizer.create_similarity_histogram(correlation_matrix, config)


def create_interactive_heatmap(correlation_matrix: pd.DataFrame,
                              title: str = 'Interactive Correlation Heatmap') -> go.Figure:
    """Convenience function for creating interactive heatmaps."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive heatmap. Install with: pip install plotly")
    visualizer = HeatmapVisualizer()
    return visualizer.create_interactive_heatmap(correlation_matrix, title)


# Test function
if __name__ == "__main__":
    print("Testing heatmap visualization module...")

    # Create sample correlation matrix
    np.random.seed(42)
    n_units = 8
    units = [f'Unit_{i+1}' for i in range(n_units)]

    # Generate sample correlation matrix
    corr_matrix = pd.DataFrame(
        np.random.uniform(-1, 1, (n_units, n_units)),
        index=units,
        columns=units
    )

    # Make it symmetric and with 1s on diagonal
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix.values, 1.0)

    print("Sample correlation matrix:")
    print(corr_matrix.round(2))

    # Test heatmap creation
    visualizer = HeatmapVisualizer()

    try:
        # Basic heatmap
        fig1 = visualizer.create_heatmap(corr_matrix)
        print("Basic heatmap created successfully")

        # Annotated heatmap
        fig2 = visualizer.create_annotated_heatmap(corr_matrix, threshold=0.3)
        print("Annotated heatmap created successfully")

        # Similarity histogram
        fig3 = visualizer.create_similarity_histogram(corr_matrix)
        print("Similarity histogram created successfully")

        # Show available colormaps
        colormaps = visualizer.get_available_colormaps()
        print(f"Available colormaps: {len(colormaps)} options")

        plt.show()

    except Exception as e:
        print(f"Error in testing: {e}")

    print("Heatmap visualization module test completed.")


# Convenience functions for backward compatibility
def create_correlation_heatmap(correlation_matrix: pd.DataFrame,
                             config: Optional[Dict[str, Any]] = None) -> plt.Figure:
    """
    Convenience function for creating correlation heatmaps.
    
    Args:
        correlation_matrix: Correlation/similarity matrix
        config: Visualization configuration
        
    Returns:
        Matplotlib figure
    """
    visualizer = HeatmapVisualizer()
    return visualizer.create_heatmap(correlation_matrix, config)


def create_interactive_heatmap(correlation_matrix: pd.DataFrame,
                             title: str = 'Interactive Correlation Heatmap',
                             config: Optional[Dict[str, Any]] = None):
    """
    Convenience function for creating interactive heatmaps.
    
    Args:
        correlation_matrix: Correlation/similarity matrix
        title: Title for the heatmap
        config: Optional configuration dict
        
    Returns:
        Plotly figure object
    """
    visualizer = HeatmapVisualizer()
    return visualizer.create_interactive_heatmap(correlation_matrix, title, config)


def create_annotated_heatmap(correlation_matrix: pd.DataFrame,
                           threshold: float = 0.5,
                           config: Optional[Dict[str, Any]] = None) -> plt.Figure:
    """
    Convenience function for creating annotated heatmaps.
    
    Args:
        correlation_matrix: Correlation matrix
        threshold: Minimum absolute correlation for annotation
        config: Visualization configuration
        
    Returns:
        Matplotlib figure
    """
    visualizer = HeatmapVisualizer()
    return visualizer.create_annotated_heatmap(correlation_matrix, threshold, config)