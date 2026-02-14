# network_visualization.py
"""
Network Visualization Module for Correlation Analysis

Provides interactive network graphs using NetworkX and Plotly, with community detection
and advanced layout algorithms for relationship visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings

# Optional imports
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("networkx not available. Network visualization will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly not available. Interactive network graphs will not be available.")

try:
    from community import community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    warnings.warn("python-louvain not available. Community detection will use basic methods.")

# Import logging
try:
    from logging_config import get_logger
    logger = get_logger("network_visualization")
except ImportError:
    import logging
    logger = logging.getLogger("network_visualization")


class NetworkVisualizer:
    """Interactive network visualization for correlation relationships."""

    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for network visualization. Install with: pip install networkx")

        self.logger = logger

        # Default configuration
        self.default_config = {
            'layout': 'spring',
            'node_size_range': (20, 50),
            'edge_width_range': (1, 8),
            'threshold': 0.3,
            'community_detection': True,
            'auto_group': True,  # New: Always detect communities automatically
            'filter_outliers': True,  # New: Filter outlier units automatically
            'min_degree': 1,  # Minimum connections required
            'min_weighted_degree': 0.5,  # Minimum correlation strength sum
            'max_isolated_ratio': 0.2,  # Max ratio of units that can be isolated
            'color_scheme': 'viridis',
            'show_labels': True,
            'label_size': 12,
            'figsize': (800, 600),
            'title': 'Correlation Network',
            # Edge coloring configuration
            'edge_coloring_enabled': True,  # Enable/disable edge coloring by correlation
            'edge_colorscale': 'Blues',  # Color scheme for edges
            'edge_color_intensity_range': (0.2, 1.0),  # Min/max color intensity
            'edge_opacity': 0.8,  # Edge transparency
            'edge_colorbar_enabled': True,  # Show colorbar for edge colors
            'edge_colorbar_title': 'Correlation Strength',
            'edge_default_color': '#888888'  # Default edge color when coloring disabled
        }
        
        # Define available color schemes for edges
        self.edge_color_schemes = {
            'Blues': 'Blues',
            'Reds': 'Reds', 
            'Greens': 'Greens',
            'Oranges': 'Oranges',
            'Purples': 'Purples',
            'Greys': 'Greys',
            'Viridis': 'Viridis',
            'Plasma': 'Plasma',
            'Inferno': 'Inferno',
            'Magma': 'Magma',
            'Cividis': 'Cividis',
            'Turbo': 'Turbo',
            'RdBu': 'RdBu',  # Red-Blue diverging
            'RdYlBu': 'RdYlBu',  # Red-Yellow-Blue diverging
            'Spectral': 'Spectral',  # Spectral
            'Coolwarm': 'RdBu_r'  # Cool-warm
        }

    def _get_edge_color_from_weight(self, weight: float, min_weight: float, max_weight: float, 
                                    colorscale: str, intensity_range: tuple, opacity: float) -> str:
        """
        Generate edge color based on correlation weight using specified color scheme.
        
        Args:
            weight: Correlation weight (absolute value)
            min_weight: Minimum weight in dataset  
            max_weight: Maximum weight in dataset
            colorscale: Color scheme name
            intensity_range: (min_intensity, max_intensity) tuple
            opacity: Color opacity (0-1)
            
        Returns:
            RGBA color string
        """
        # Normalize weight to 0-1 range
        if max_weight > min_weight:
            normalized_weight = (weight - min_weight) / (max_weight - min_weight)
        else:
            normalized_weight = 0.5
            
        # Map to intensity range
        min_intensity, max_intensity = intensity_range
        intensity = min_intensity + (max_intensity - min_intensity) * normalized_weight
        
        # Generate color based on scheme
        if colorscale in ['Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'Greys']:
            # Single-hue schemes
            if colorscale == 'Blues':
                return f'rgba(0, 0, {int(255 * intensity)}, {opacity})'
            elif colorscale == 'Reds':
                return f'rgba({int(255 * intensity)}, 0, 0, {opacity})'
            elif colorscale == 'Greens':
                return f'rgba(0, {int(255 * intensity)}, 0, {opacity})'
            elif colorscale == 'Oranges':
                return f'rgba({int(255 * intensity)}, {int(165 * intensity)}, 0, {opacity})'
            elif colorscale == 'Purples':
                return f'rgba({int(128 * intensity)}, 0, {int(128 * intensity)}, {opacity})'
            elif colorscale == 'Greys':
                gray_val = int(255 * intensity)
                return f'rgba({gray_val}, {gray_val}, {gray_val}, {opacity})'
        elif colorscale in ['Viridis', 'Plasma', 'Inferno', 'Magma']:
            # Perceptually uniform schemes - approximate with RGB
            if colorscale == 'Viridis':
                r = int(255 * (0.267004 + intensity * 0.686))
                g = int(255 * intensity)
                b = int(255 * (0.329415 + intensity * 0.670))
                return f'rgba({r}, {g}, {b}, {opacity})'
            elif colorscale == 'Plasma':
                r = int(255 * (0.050383 + intensity * 0.950))
                g = int(255 * intensity * 0.7)
                b = int(255 * (0.505495 + intensity * 0.494))
                return f'rgba({r}, {g}, {b}, {opacity})'
        
        # Default fallback to blue scheme
        return f'rgba(0, 0, {int(255 * intensity)}, {opacity})'

    def _create_colored_edge_traces(self, graph, positions: dict, config: dict) -> list:
        """
        Create edge traces with color mapping based on correlation strength.
        
        Args:
            graph: NetworkX graph
            positions: Node positions dict
            config: Configuration dict
            
        Returns:
            List of edge traces and optional colorbar trace
        """
        edge_traces = []
        
        if not config.get('edge_coloring_enabled', True):
            # Create simple edges without coloring
            edge_x, edge_y = [], []
            for edge in graph.edges():
                x0, y0 = positions[edge[0]]
                x1, y1 = positions[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=2, color=config.get('edge_default_color', '#888888')),
                hoverinfo='none',
                showlegend=False
            )
            return [edge_trace]
        
        # Collect all weights for normalization
        all_weights = []
        for edge in graph.edges(data=True):
            weight = edge[2].get('weight', 0)
            abs_weight = abs(weight)
            all_weights.append(abs_weight)
        
        if not all_weights:
            return []
            
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        
        # Get color configuration
        colorscale = config.get('edge_colorscale', 'Blues')
        intensity_range = config.get('edge_color_intensity_range', (0.2, 1.0))
        opacity = config.get('edge_opacity', 0.8)
        
        # Create individual edge traces with colors
        for edge in graph.edges(data=True):
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            weight = edge[2].get('weight', 0)
            abs_weight = abs(weight)
            
            # Get color for this edge
            edge_color = self._get_edge_color_from_weight(
                abs_weight, min_weight, max_weight, colorscale, intensity_range, opacity
            )
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=2, color=edge_color),
                hoverinfo='text',
                text=f'Correlation: {weight:.3f}',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Add colorbar if enabled
        if config.get('edge_colorbar_enabled', True) and all_weights:
            colorbar_trace = go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    color=[min_weight, max_weight],
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(
                        title=config.get('edge_colorbar_title', 'Correlation Strength'),
                        thickness=15,
                        xanchor='left'
                    ),
                    size=0.1,
                    opacity=0.01
                ),
                showlegend=False,
                hoverinfo='skip'
            )
            edge_traces.append(colorbar_trace)
            
        return edge_traces

    def get_available_edge_colorschemes(self) -> dict:
        """
        Get all available edge color schemes.
        
        Returns:
            Dictionary of color scheme names and descriptions
        """
        return {
            'Blues': 'Blue gradient (light to dark blue)',
            'Reds': 'Red gradient (light to dark red)', 
            'Greens': 'Green gradient (light to dark green)',
            'Oranges': 'Orange gradient (light to dark orange)',
            'Purples': 'Purple gradient (light to dark purple)',
            'Greys': 'Grey gradient (light to dark grey)',
            'Viridis': 'Viridis (perceptually uniform)',
            'Plasma': 'Plasma (perceptually uniform)',
            'Inferno': 'Inferno (perceptually uniform)',
            'Magma': 'Magma (perceptually uniform)',
            'Cividis': 'Cividis (colorblind friendly)',
            'Turbo': 'Turbo (high contrast)',
            'RdBu': 'Red-Blue diverging',
            'RdYlBu': 'Red-Yellow-Blue diverging',
            'Spectral': 'Spectral colors',
            'Coolwarm': 'Cool-warm colors'
        }

    def set_edge_color_config(self, config_updates: dict) -> dict:
        """
        Update edge color configuration and return merged config.
        
        Args:
            config_updates: Dictionary with edge color configuration updates
            
        Returns:
            Updated configuration dictionary
        """
        # Validate colorscale
        if 'edge_colorscale' in config_updates:
            if config_updates['edge_colorscale'] not in self.edge_color_schemes:
                available = list(self.edge_color_schemes.keys())
                raise ValueError(f"Invalid colorscale. Available options: {available}")
        
        # Validate intensity range
        if 'edge_color_intensity_range' in config_updates:
            range_val = config_updates['edge_color_intensity_range']
            if not isinstance(range_val, (list, tuple)) or len(range_val) != 2:
                raise ValueError("edge_color_intensity_range must be a tuple/list of 2 values")
            if not (0 <= range_val[0] <= 1 and 0 <= range_val[1] <= 1):
                raise ValueError("edge_color_intensity_range values must be between 0 and 1")
            if range_val[0] >= range_val[1]:
                raise ValueError("First value must be less than second value in intensity range")
        
        # Validate opacity
        if 'edge_opacity' in config_updates:
            opacity = config_updates['edge_opacity']
            if not (0 <= opacity <= 1):
                raise ValueError("edge_opacity must be between 0 and 1")
        
        # Merge with defaults
        updated_config = self.default_config.copy()
        updated_config.update(config_updates)
        
        return updated_config

    def create_network_from_correlation(self, correlation_matrix: pd.DataFrame,
                                         config: Optional[Dict[str, Any]] = None):
        """
        Create NetworkX graph from correlation matrix with optional outlier filtering.

        Args:
            correlation_matrix: Correlation/similarity matrix
            config: Network configuration

        Returns:
            Dict containing graph and filtering report, or just NetworkX graph for backward compatibility
        """
        try:
            self.logger.info("Creating network from correlation matrix")

            if not NETWORKX_AVAILABLE:
                raise ImportError("NetworkX is required for network visualization")

            if correlation_matrix.empty:
                raise ValueError("Correlation matrix is empty")

            # Merge configurations
            net_config = self.default_config.copy()
            if config:
                net_config.update(config)

            # Create initial graph
            G = nx.Graph()

            # Add nodes
            for unit in correlation_matrix.index:
                G.add_node(unit, name=unit)

            # Add edges based on threshold
            threshold = net_config.get('threshold', 0.3)

            for i, unit1 in enumerate(correlation_matrix.index):
                for j, unit2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicate edges
                        correlation = correlation_matrix.loc[unit1, unit2]
                        if abs(correlation) >= threshold:
                            G.add_edge(unit1, unit2, weight=correlation, abs_weight=abs(correlation))

            self.logger.info(f"Initial network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

            # Apply top K connections filtering if enabled
            if net_config.get('top_k_connections', False):
                k_value = net_config.get('k_value', 5)
                G = self.filter_top_k_connections(G, k_value)
                self.logger.info(f"Applied top K={k_value} connections filtering")

            # Apply outlier filtering if enabled
            filtering_report = None
            if net_config.get('filter_outliers', True):
                filter_result = self.filter_outlier_units(
                    G, correlation_matrix,
                    min_degree=net_config.get('min_degree', 1),
                    min_weighted_degree=net_config.get('min_weighted_degree', 0.5),
                    max_isolated_ratio=net_config.get('max_isolated_ratio', 0.2)
                )
                G = filter_result['filtered_graph']
                filtering_report = filter_result['filtering_report']

                if filtering_report and filtering_report.get('removed_nodes', 0) > 0:
                    self.logger.info(f"Filtered {filtering_report['removed_nodes']} outlier units")

            # Return graph and filtering report for new functionality,
            # or just graph for backward compatibility
            if filtering_report:
                return {
                    'graph': G,
                    'filtering_report': filtering_report
                }
            else:
                return G

        except Exception as e:
            error_msg = f"Error creating network: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def detect_communities(self, graph, method: str = 'auto') -> Dict[str, int]:
        """
        Detect communities in the network using automatic method selection.

        Args:
            graph: NetworkX graph
            method: Community detection method ('auto', 'louvain', 'greedy', 'label_propagation')
                   'auto' will try Louvain first, then fallbacks

        Returns:
            Dictionary mapping node to community ID
        """
        try:
            # Auto method selection - always try Louvain first for best results
            if method == 'auto':
                if LOUVAIN_AVAILABLE:
                    method = 'louvain'
                    self.logger.info("Auto-selecting Louvain method for community detection")
                else:
                    method = 'greedy'  # Best NetworkX fallback
                    self.logger.info("Auto-selecting greedy method (Louvain not available)")

            self.logger.info(f"Detecting communities using {method} method")

            if method == 'louvain' and LOUVAIN_AVAILABLE:
                partition = community_louvain.best_partition(graph)
            elif method == 'greedy':
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(graph))
                partition = {}
                for i, community in enumerate(communities):
                    for node in community:
                        partition[node] = i
            elif method == 'label_propagation':
                from networkx.algorithms.community import label_propagation_communities
                communities = list(label_propagation_communities(graph))
                partition = {}
                for i, community in enumerate(communities):
                    for node in community:
                        partition[node] = i
            else:
                # Fallback: assign all nodes to same community
                self.logger.warning(f"Community detection method '{method}' not available, using single community")
                partition = {node: 0 for node in graph.nodes()}

            n_communities = len(set(partition.values()))
            self.logger.info(f"Detected {n_communities} communities using {method} method")
            return partition

        except Exception as e:
            self.logger.error(f"Error in community detection: {e}")
            # Return single community as fallback
            return {node: 0 for node in graph.nodes()}

    def calculate_node_positions(self, graph, layout: str = 'spring',
                                **layout_kwargs) -> Dict[str, Tuple[float, float]]:
        """
        Calculate node positions using specified layout algorithm.

        Args:
            graph: NetworkX graph
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'random')
            layout_kwargs: Additional arguments for layout algorithm

        Returns:
            Dictionary of node positions
        """
        try:
            self.logger.info(f"Calculating node positions using {layout} layout")

            # Set default parameters for each layout
            if layout == 'spring':
                pos = nx.spring_layout(graph, k=1/np.sqrt(len(graph)), iterations=50, **layout_kwargs)
            elif layout == 'spring_optimized':
                # Optimized spring layout for large networks
                pos = nx.spring_layout(graph, k=0.5/np.sqrt(len(graph)), iterations=100, **layout_kwargs)
            elif layout == 'circular':
                pos = nx.circular_layout(graph, **layout_kwargs)
            elif layout == 'kamada_kawai':
                try:
                    # Handle negative weights by using absolute values
                    graph_copy = graph.copy()
                    for u, v, d in graph_copy.edges(data=True):
                        if 'weight' in d and d['weight'] < 0:
                            d['weight'] = abs(d['weight'])
                    pos = nx.kamada_kawai_layout(graph_copy, **layout_kwargs)
                except Exception as kamada_error:
                    self.logger.warning(f"Kamada-Kawai layout failed: {kamada_error}, falling back to spring layout")
                    pos = nx.spring_layout(graph, **layout_kwargs)
            elif layout == 'random':
                pos = nx.random_layout(graph, **layout_kwargs)
            elif layout == 'spectral':
                pos = nx.spectral_layout(graph, **layout_kwargs)
            else:
                self.logger.warning(f"Unknown layout '{layout}', using spring layout")
                pos = nx.spring_layout(graph, **layout_kwargs)

            return pos

        except Exception as e:
            self.logger.error(f"Error calculating positions: {e}")
            # Fallback to circular layout
            return nx.circular_layout(graph)

    def create_plotly_network(self, graph,
                             positions: Optional[Dict[str, Tuple[float, float]]] = None,
                             communities: Optional[Dict[str, int]] = None,
                             config: Optional[Dict[str, Any]] = None):
        """
        Create interactive Plotly network visualization.

        Args:
            graph: NetworkX graph
            positions: Node positions (calculated if None)
            communities: Community assignments (detected if None)
            config: Visualization configuration

        Returns:
            Plotly figure
        """
        try:
            if not PLOTLY_AVAILABLE:
                raise ImportError("Plotly is required for interactive network visualization")

            self.logger.info("Creating Plotly network visualization")

            # Merge configurations
            viz_config = self.default_config.copy()
            if config:
                viz_config.update(config)

            # Calculate positions if not provided
            if positions is None:
                positions = self.calculate_node_positions(graph, viz_config.get('layout', 'spring'))

            # Detect communities automatically if not provided and auto_group is enabled
            if communities is None:
                auto_group = config.get('auto_group', True) if config else True
                if auto_group:
                    communities = self.detect_communities(graph, method='auto')

            # Prepare node data
            node_x, node_y, node_text, node_color, node_size = [], [], [], [], []

            # Check if user-defined groups are available
            user_groups = config.get('groups', {}) if config else {}
            user_group_colors = config.get('group_colors', {}) if config else {}

            for node in graph.nodes():
                x, y = positions[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))

                # Node color based on user-defined groups or communities
                if user_groups and str(node) in user_groups:
                    # Use user-defined group color
                    group_name = user_groups[str(node)]
                    if group_name in user_group_colors:
                        node_color.append(user_group_colors[group_name])
                    else:
                        # Fallback to community-based coloring
                        node_color.append(communities.get(node, 0) if communities else 0)
                elif communities:
                    # Use community-based coloring
                    node_color.append(communities.get(node, 0))
                else:
                    node_color.append(0)

                # Node size based on degree (number of connections)
                degree = graph.degree(node)
                size_range = viz_config['node_size_range']
                max_degree = max(dict(graph.degree()).values()) if graph.degree() else 1
                normalized_size = np.interp(degree, [0, max_degree], size_range)
                node_size.append(normalized_size)

            # Create edge traces with color mapping based on correlation strength
            edge_traces = self._create_colored_edge_traces(graph, positions, viz_config)

            # Create node trace
            # Determine if we should show color scale (only for community-based coloring)
            show_color_scale = not user_groups

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text' if viz_config.get('show_labels', True) else 'markers',
                hoverinfo='text',
                text=node_text if viz_config.get('show_labels', True) else None,
                textposition="top center",
                textfont=dict(size=viz_config.get('label_size', 12)),
                marker=dict(
                    showscale=show_color_scale,
                    colorscale=viz_config.get('color_scheme', 'viridis') if show_color_scale else None,
                    reversescale=False,
                    color=node_color,
                    size=node_size,
                    colorbar=dict(
                        thickness=15,
                        title='Groups' if user_groups else 'Community',
                        xanchor='left'
                    ) if show_color_scale else None,
                    line_width=2))

            # Create figure
            title_text = viz_config.get('title', 'Correlation Network')
            if user_groups:
                title_text += f" ({len(set(user_groups.values()))} Groups)"
            elif communities:
                n_communities = len(set(communities.values()))
                title_text += f" - {n_communities} Communities Detected"

            # Combine traces - add all edge traces plus node trace
            all_traces = edge_traces + [node_trace]

            # Use fullscreen layout configuration if available
            if 'figure_size_mode' in viz_config:
                layout_config = self.get_fullscreen_layout_config(viz_config)
                # Override title with dynamic content
                layout_config['title']['text'] = title_text
            else:
                # Fallback to original layout
                layout_config = go.Layout(
                    title=dict(
                        text=title_text,
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    width=viz_config.get('figsize', (800, 600))[0],
                    height=viz_config.get('figsize', (800, 600))[1]
                )

            fig = go.Figure(data=all_traces, layout=layout_config)

            # Show the plot with fullscreen configuration if available
            if 'figure_size_mode' in viz_config:
                plotly_config = self.get_plotly_config_for_fullscreen(viz_config)
                fig.show(config=plotly_config)
                self.logger.info("Plotly network displayed with fullscreen configuration")
            else:
                fig.show()
                self.logger.info("Plotly network displayed with standard configuration")

            self.logger.info("Plotly network visualization created successfully")
            return fig

        except Exception as e:
            error_msg = f"Error creating Plotly network: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def create_matplotlib_network(self, graph,
                                 positions: Optional[Dict[str, Tuple[float, float]]] = None,
                                 communities: Optional[Dict[str, int]] = None,
                                 config: Optional[Dict[str, Any]] = None):
        """
        Create static matplotlib network visualization.

        Args:
            graph: NetworkX graph
            positions: Node positions
            communities: Community assignments
            config: Visualization configuration

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            self.logger.info("Creating matplotlib network visualization")

            # Merge configurations
            viz_config = self.default_config.copy()
            if config:
                viz_config.update(config)

            # Calculate positions if not provided
            if positions is None:
                positions = self.calculate_node_positions(graph, viz_config.get('layout', 'spring'))

            # Detect communities automatically if not provided and auto_group is enabled
            if communities is None:
                auto_group = config.get('auto_group', True) if config else True
                if auto_group:
                    communities = self.detect_communities(graph, method='auto')

            # Create figure
            fig, ax = plt.subplots(figsize=(viz_config['figsize'][0]/100, viz_config['figsize'][1]/100))

            # Draw edges
            edges = list(graph.edges())
            if edges:
                nx.draw_networkx_edges(
                    graph, positions, ax=ax,
                    width=[graph[u][v].get('abs_weight', 1.0) * 2 for u, v in edges],
                    alpha=0.7, edge_color='gray'
                )

            # Draw nodes - always color by communities (automatic detection)
            if communities:
                # Color by community with distinct colors
                community_ids = sorted(list(set(communities.values())))
                n_communities = len(community_ids)

                # Use a colormap that provides distinct colors
                if n_communities <= 10:
                    # Use tab10 for up to 10 distinct colors
                    base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
                elif n_communities <= 20:
                    # Use tab20 for up to 20 distinct colors
                    base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
                else:
                    # Use viridis for more communities
                    base_colors = plt.cm.viridis(np.linspace(0, 1, n_communities))

                # Create color map
                color_map = {comm_id: base_colors[i % len(base_colors)] for i, comm_id in enumerate(community_ids)}
                node_colors = [color_map[communities.get(node, 0)] for node in graph.nodes()]

                # Add community legend
                legend_elements = []
                for comm_id in community_ids:
                    color = color_map[comm_id]
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                    markerfacecolor=color, markersize=10,
                                                    label=f'Community {comm_id}'))

                ax.legend(handles=legend_elements, title="Communities", bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                node_colors = 'lightblue'

            # Node sizes based on degree
            degrees = dict(graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            node_sizes = [degrees[node] / max_degree * 500 + 200 for node in graph.nodes()]

            nx.draw_networkx_nodes(
                graph, positions, ax=ax,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.8
            )

            # Draw labels
            if viz_config.get('show_labels', True):
                nx.draw_networkx_labels(
                    graph, positions, ax=ax,
                    font_size=viz_config.get('label_size', 12),
                    font_weight='bold'
                )

            # Customize title with community information
            title = viz_config.get('title', 'Correlation Network')
            if communities:
                n_communities = len(set(communities.values()))
                title += f" - {n_communities} Communities Detected"

            ax.set_title(title, fontsize=16, pad=20)
            ax.axis('off')
            plt.tight_layout()

            self.logger.info("Matplotlib network visualization created successfully")
            return fig

        except Exception as e:
            error_msg = f"Error creating matplotlib network: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def export_network_html(self, fig: go.Figure, filepath: str) -> bool:
        """
        Export Plotly network to HTML file.

        Args:
            fig: Plotly figure
            filepath: Output HTML file path

        Returns:
            True if successful
        """
        try:
            self.logger.info(f"Exporting network to HTML: {filepath}")

            if not PLOTLY_AVAILABLE:
                raise ImportError("Plotly is required for HTML export")

            fig.write_html(filepath, include_plotlyjs='cdn', full_html=True)

            self.logger.info(f"Network exported successfully to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting network HTML: {e}")
            return False

    def filter_outlier_units(self, graph, correlation_matrix: pd.DataFrame,
                           min_degree: int = 1, min_weighted_degree: float = 0.0,
                           max_isolated_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Filter out outlier units with low connectivity.

        Args:
            graph: NetworkX graph
            correlation_matrix: Original correlation matrix
            min_degree: Minimum number of connections required
            min_weighted_degree: Minimum sum of absolute correlations
            max_isolated_ratio: Maximum ratio of units that can be isolated

        Returns:
            Dict with filtered graph and filtering report
        """
        try:
            self.logger.info("Filtering outlier units from network")

            # Calculate connectivity metrics for each node
            node_metrics = {}
            for node in graph.nodes():
                degree = graph.degree(node)
                # Weighted degree (sum of absolute correlation strengths)
                weighted_degree = sum(abs(edge[2].get('weight', 0))
                                    for edge in graph.edges(node, data=True))

                node_metrics[node] = {
                    'degree': degree,
                    'weighted_degree': weighted_degree,
                    'is_isolated': degree == 0
                }

            # Identify outliers
            outliers = []
            kept_units = []

            for node, metrics in node_metrics.items():
                # Check if node meets minimum connectivity criteria
                meets_degree = metrics['degree'] >= min_degree
                meets_weighted = metrics['weighted_degree'] >= min_weighted_degree

                if not (meets_degree and meets_weighted):
                    outliers.append((node, metrics))
                else:
                    kept_units.append(node)

            # Apply maximum isolated ratio constraint
            total_units = len(graph.nodes())
            max_allowed_outliers = int(total_units * max_isolated_ratio)

            if len(outliers) > max_allowed_outliers:
                # Sort outliers by connectivity (keep the most connected ones)
                outliers.sort(key=lambda x: (x[1]['degree'], x[1]['weighted_degree']), reverse=True)
                # Keep only the most connected outliers up to the limit
                kept_outliers = outliers[:max_allowed_outliers]
                removed_outliers = outliers[max_allowed_outliers:]

                # Add kept outliers back to kept_units
                for outlier, _ in kept_outliers:
                    kept_units.append(outlier)
                    outliers.remove((outlier, _))
            else:
                removed_outliers = []

            # Create filtered graph
            filtered_graph = graph.subgraph(kept_units).copy()

            # Create filtering report
            filtering_report = {
                'original_nodes': total_units,
                'filtered_nodes': len(kept_units),
                'removed_nodes': len(outliers),
                'removed_units': [unit for unit, _ in outliers],
                'kept_units': kept_units,
                'filtering_criteria': {
                    'min_degree': min_degree,
                    'min_weighted_degree': min_weighted_degree,
                    'max_isolated_ratio': max_isolated_ratio
                },
                'node_metrics': node_metrics
            }

            self.logger.info(f"Filtered {len(outliers)} outlier units from network")

            return {
                'filtered_graph': filtered_graph,
                'filtering_report': filtering_report
            }

        except Exception as e:
            self.logger.error(f"Error filtering outlier units: {e}")
            return {
                'filtered_graph': graph,
                'filtering_report': {
                    'error': str(e),
                    'original_nodes': graph.number_of_nodes(),
                    'filtered_nodes': graph.number_of_nodes(),
                    'removed_nodes': 0
                }
            }

    def filter_top_k_connections(self, graph, k: int = 5):
        """
        Filter graph to keep only the top K strongest connections per node.

        Args:
            graph: NetworkX graph
            k: Number of top connections to keep per node

        Returns:
            Filtered NetworkX graph
        """
        try:
            self.logger.info(f"Filtering to top K={k} connections per node")

            # Create a new graph to store filtered edges
            filtered_graph = nx.Graph()

            # Add all nodes
            for node in graph.nodes():
                filtered_graph.add_node(node, **graph.nodes[node])

            # For each node, find its top K strongest connections
            edges_to_keep = set()

            for node in graph.nodes():
                # Get all edges connected to this node
                node_edges = []
                for neighbor in graph.neighbors(node):
                    weight = graph[node][neighbor].get('weight', 0)
                    abs_weight = abs(weight)
                    node_edges.append((node, neighbor, weight, abs_weight))

                # Sort by absolute weight (strongest first)
                node_edges.sort(key=lambda x: x[3], reverse=True)

                # Keep only top K edges
                for i, (u, v, weight, abs_weight) in enumerate(node_edges):
                    if i < k:
                        # Add edge (avoid duplicates by using frozenset)
                        edge_key = frozenset([u, v])
                        if edge_key not in edges_to_keep:
                            edges_to_keep.add(edge_key)
                            filtered_graph.add_edge(u, v, weight=weight, abs_weight=abs_weight)

            self.logger.info(f"Filtered graph: {filtered_graph.number_of_nodes()} nodes, {filtered_graph.number_of_edges()} edges (top {k} per node)")
            return filtered_graph

        except Exception as e:
            self.logger.error(f"Error filtering top K connections: {e}")
            return graph  # Return original graph on error

    def get_network_statistics(self, graph) -> Dict[str, Any]:
        """Calculate network statistics."""
        try:
            stats = {
                'n_nodes': graph.number_of_nodes(),
                'n_edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'average_clustering': nx.average_clustering(graph),
                'connected_components': nx.number_connected_components(graph)
            }

            # Add degree statistics
            degrees = [d for n, d in graph.degree()]
            if degrees:
                stats.update({
                    'avg_degree': np.mean(degrees),
                    'max_degree': max(degrees),
                    'min_degree': min(degrees)
                })

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating network statistics: {e}")
            return {}

    def get_fullscreen_layout_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized layout configuration for fullscreen display."""
        try:
            # Get figure dimensions
            figure_dimensions = config.get('figure_dimensions', (1600, 1000))
            width, height = figure_dimensions
            
            # Determine if this is a fullscreen mode
            figure_size_mode = config.get('figure_size_mode', 'large')
            is_fullscreen = figure_size_mode in ['fullscreen', 'auto-detect'] or width >= 2000
            
            # Base layout config
            layout_config = {
                'title': dict(
                    text=config.get('title', 'Network Visualization'),
                    font=dict(size=20 if is_fullscreen else 16),
                    x=0.5,  # Center title
                    xanchor='center'
                ),
                'showlegend': config.get('show_legend', False),
                'hovermode': 'closest',
                'width': width,
                'height': height,
                'xaxis': dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    fixedrange=not config.get('enable_zoom_pan', True)
                ),
                'yaxis': dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    fixedrange=not config.get('enable_zoom_pan', True)
                )
            }
            
            # Margin configuration based on fullscreen settings
            if config.get('minimize_margins', True) and is_fullscreen:
                # Minimal margins for maximum space usage
                layout_config['margin'] = dict(l=5, r=5, t=40, b=5, pad=0)
            elif config.get('browser_fullscreen', True):
                # Optimized for browser viewing
                layout_config['margin'] = dict(l=20, r=20, t=60, b=20, pad=5)
            else:
                # Standard margins
                layout_config['margin'] = dict(l=40, r=40, t=80, b=40, pad=10)
            
            # Responsive layout settings
            if config.get('responsive_layout', True):
                layout_config['autosize'] = True
            
            # Enhanced interactivity for fullscreen
            if is_fullscreen:
                layout_config['dragmode'] = 'pan'  # Default to pan mode for large networks
                layout_config['hovermode'] = 'closest'
                
            # Background styling for better fullscreen appearance
            if config.get('browser_fullscreen', True):
                layout_config['plot_bgcolor'] = 'rgba(0,0,0,0)'  # Transparent background
                layout_config['paper_bgcolor'] = 'white'  # White paper background
                
            self.logger.info(f"Generated fullscreen layout config: {width}x{height}, fullscreen={is_fullscreen}")
            return layout_config
            
        except Exception as e:
            self.logger.warning(f"Error generating fullscreen layout config: {e}")
            # Return basic config as fallback
            return {
                'title': config.get('title', 'Network Visualization'),
                'showlegend': False,
                'hovermode': 'closest',
                'margin': dict(b=20, l=5, r=5, t=40),
                'width': config.get('figure_dimensions', (1600, 1000))[0],
                'height': config.get('figure_dimensions', (1600, 1000))[1]
            }

    def get_plotly_config_for_fullscreen(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get Plotly configuration optimized for fullscreen display."""
        figure_size_mode = config.get('figure_size_mode', 'large')
        is_fullscreen = figure_size_mode in ['fullscreen', 'auto-detect', 'browser_fullscreen']
        
        plotly_config = {
            'displayModeBar': not config.get('hide_toolbar', False),
            'autosizable': True,
            'fillFrame': True,  # Fill the entire frame
            'frameMargins': 0 if config.get('minimize_margins', True) else 0.1,
        }
        
        if is_fullscreen:
            # Enhanced config for fullscreen
            plotly_config.update({
                'displaylogo': False,  # Hide Plotly logo
                'modeBarButtonsToRemove': [
                    'select2d', 'lasso2d', 'autoScale2d', 'hoverClosestCartesian',
                    'hoverCompareCartesian', 'toggleSpikelines'
                ] if config.get('hide_toolbar', False) else [],
                'modeBarButtonsToAdd': [
                    'pan2d', 'zoom2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'
                ],
                'scrollZoom': True,  # Enable scroll to zoom
                'doubleClick': 'reset',  # Double click to reset zoom
                'showTips': False,  # Hide tips for cleaner look
                'watermark': False,  # Remove watermark
            })
        
        return plotly_config


# Convenience functions
def create_correlation_network(correlation_matrix,
                              config: Optional[Dict[str, Any]] = None):
    """Convenience function for creating correlation networks."""
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX is required for network visualization. Install with: pip install networkx")
    visualizer = NetworkVisualizer()
    result = visualizer.create_network_from_correlation(correlation_matrix, config)

    # Handle new return format (dict) or old format (graph only)
    if isinstance(result, dict):
        return result['graph']  # Return graph for backward compatibility
    else:
        return result


def create_interactive_network(graph,
                             config: Optional[Dict[str, Any]] = None):
    """Convenience function for interactive network visualization."""
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX is required for network visualization. Install with: pip install networkx")
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive network visualization. Install with: pip install plotly")
    visualizer = NetworkVisualizer()
    return visualizer.create_plotly_network(graph, config=config)


def create_static_network(graph,
                         config: Optional[Dict[str, Any]] = None):
    """Convenience function for static network visualization."""
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX is required for network visualization. Install with: pip install networkx")
    visualizer = NetworkVisualizer()
    return visualizer.create_matplotlib_network(graph, config=config)


def detect_network_communities(graph, method: str = 'louvain') -> Dict[str, int]:
    """Convenience function for community detection."""
    visualizer = NetworkVisualizer()
    return visualizer.detect_communities(graph, method)


def create_hierarchical_network(correlation_matrix, communities: Dict[str, int],
                               config: Optional[Dict[str, Any]] = None):
    """Convenience function for hierarchical network visualization."""
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX is required for hierarchical network visualization. Install with: pip install networkx")
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for hierarchical network visualization. Install with: pip install plotly")
    visualizer = HierarchicalNetworkVisualizer()
    return visualizer.create_hierarchical_network(correlation_matrix, communities, config)


# Hierarchical Network Visualization
class HierarchicalNetworkVisualizer(NetworkVisualizer):
    """
    Hierarchical network visualization with drill-down capabilities.
    Shows communities at different levels and allows exploration.
    """

    def __init__(self):
        super().__init__()
        self.current_level = 'communities'  # 'communities' or 'detailed'
        self.selected_community = None
        self.original_graph = None
        self.original_communities = None
        self.original_positions = None

    def create_hierarchical_network(self, graph, communities: Dict[str, int],
                                   config: Optional[Dict[str, Any]] = None):
        """
        Create hierarchical network visualization with community aggregation and drill-down.

        Args:
            graph: NetworkX graph
            communities: Community assignments
            config: Visualization configuration

        Returns:
            Plotly figure with hierarchical view
        """
        try:
            if not PLOTLY_AVAILABLE:
                raise ImportError("Plotly is required for hierarchical network visualization")

            self.logger.info("Creating hierarchical network visualization")

            # Store original data for drill-down
            self.original_graph = graph
            self.original_communities = communities.copy()
            self.original_positions = self.calculate_node_positions(graph, 'spring')

            # Merge configurations
            viz_config = self.default_config.copy()
            if config:
                viz_config.update(config)

            # Reset to community level view
            self.current_level = 'communities'
            self.selected_community = None

            # Create initial community-level visualization
            fig = self._create_community_level_view(viz_config)

            self.logger.info("Hierarchical network visualization created successfully")
            return fig

        except Exception as e:
            error_msg = f"Error creating hierarchical network: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _create_community_level_view(self, config):
        """Create the community-level view with drilldown capabilities."""
        # Group nodes by community
        community_nodes = {}
        for node, comm_id in self.original_communities.items():
            if comm_id not in community_nodes:
                community_nodes[comm_id] = []
            community_nodes[comm_id].append(node)

        # Create all traces (community overview + individual community details)
        all_traces = []
        community_ids = sorted(community_nodes.keys())
        
        # Create community overview traces
        overview_traces = self._create_overview_traces(community_nodes, config)
        all_traces.extend(overview_traces)
        
        # Create detailed traces for each community
        detailed_traces_by_community = {}
        for comm_id in community_ids:
            detailed_traces = self._create_detailed_community_traces(comm_id, community_nodes[comm_id], config)
            detailed_traces_by_community[comm_id] = detailed_traces
            all_traces.extend(detailed_traces)
        
        # Create buttons with proper visibility toggling
        buttons = []
        
        # Overview button - show overview traces, hide all detailed traces
        overview_visibility = [True] * len(overview_traces)
        for detailed_traces in detailed_traces_by_community.values():
            overview_visibility.extend([False] * len(detailed_traces))
            
        buttons.append(dict(
            label="Overview",
            method="update",
            args=[{"visible": overview_visibility},
                  {"title.text": 'Hierarchical Network View - Communities<br><i>Click buttons below to explore individual communities</i>'}]
        ))
        
        # Community buttons - hide overview, show specific community, hide others
        for comm_id in community_ids:
            community_visibility = [False] * len(overview_traces)  # Hide overview
            
            for other_comm_id, detailed_traces in detailed_traces_by_community.items():
                if other_comm_id == comm_id:
                    community_visibility.extend([True] * len(detailed_traces))  # Show this community
                else:
                    community_visibility.extend([False] * len(detailed_traces))  # Hide other communities
            
            community_size = len(community_nodes[comm_id])
            buttons.append(dict(
                label=f"Community {comm_id} ({community_size} units)",
                method="update",
                args=[{"visible": community_visibility},
                      {"title.text": f'Community {comm_id} - Detailed View<br><i>Showing {community_size} units and their connections</i>'}]
            ))

        # Create figure with all traces
        fig = go.Figure(data=all_traces,
                       layout=go.Layout(
                           title=dict(
                               text='Hierarchical Network View - Communities<br><i>Click buttons below to explore individual communities</i>',
                               font=dict(size=16)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=80),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=config.get('figsize', (800, 600))[0],
                           height=config.get('figsize', (800, 600))[1],
                           updatemenus=[dict(
                               type="buttons",
                               direction="left",
                               buttons=buttons,
                               pad={"r": 10, "t": 10},
                               showactive=True,
                               x=0,
                               xanchor="left",
                               y=1.02,
                               yanchor="bottom"
                           )]
                       ))

        # Set initial visibility (overview only)
        for i, trace in enumerate(fig.data):
            if i < len(overview_traces):
                trace.visible = True
            else:
                trace.visible = False

        return fig

    def _create_overview_traces(self, community_nodes, config):
        """Create traces for community overview."""
        community_sizes = {comm_id: len(nodes) for comm_id, nodes in community_nodes.items()}
        
        # Create meta-graph (communities as nodes)
        meta_graph = nx.Graph()
        community_positions = {}

        # Calculate community properties and positions
        for comm_id, nodes in community_nodes.items():
            subgraph = self.original_graph.subgraph(nodes)
            
            # Community position as average of node positions
            node_positions = [self.original_positions[node] for node in nodes]
            avg_x = sum(pos[0] for pos in node_positions) / len(node_positions)
            avg_y = sum(pos[1] for pos in node_positions) / len(node_positions)
            community_positions[comm_id] = (avg_x, avg_y)
            
            # Add community node to meta-graph
            meta_graph.add_node(comm_id, size=len(nodes), density=nx.density(subgraph))

        # Add edges between communities based on inter-community connections
        for comm1_id, nodes1 in community_nodes.items():
            for comm2_id, nodes2 in community_nodes.items():
                if comm1_id < comm2_id:  # Avoid duplicates
                    # Count connections between communities
                    inter_connections = 0
                    total_weight = 0
                    
                    for node1 in nodes1:
                        for node2 in nodes2:
                            if self.original_graph.has_edge(node1, node2):
                                inter_connections += 1
                                total_weight += abs(self.original_graph[node1][node2].get('weight', 1))
                    
                    if inter_connections > 0:
                        avg_weight = total_weight / inter_connections
                        meta_graph.add_edge(comm1_id, comm2_id, weight=avg_weight, connections=inter_connections)

        # Create edge traces for meta-graph
        edge_traces = self._create_colored_edge_traces(meta_graph, community_positions, config)
        
        # Create node trace for communities
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        for comm_id in meta_graph.nodes():
            x, y = community_positions[comm_id]
            node_x.append(x)
            node_y.append(y)
            
            size = community_sizes[comm_id]
            node_text.append(f'Community {comm_id}<br>{size} units')
            node_color.append(comm_id)
            
            # Size based on community size
            size_range = config.get('node_size_range', (30, 80))
            max_size = max(community_sizes.values())
            normalized_size = np.interp(size, [1, max_size], size_range)
            node_size.append(normalized_size)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[f'Community {comm_id}' for comm_id in meta_graph.nodes()],
            textposition="middle center",
            textfont=dict(size=config.get('label_size', 12), color='white'),
            marker=dict(
                showscale=True,
                colorscale='viridis',
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Community ID',
                    xanchor='left'
                ),
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        return edge_traces + [node_trace]

    def _create_detailed_community_traces(self, comm_id, community_nodes_list, config):
        """Create traces for detailed community view."""
        # Create subgraph for this community
        subgraph = self.original_graph.subgraph(community_nodes_list).copy()
        
        # Get positions for community nodes
        community_positions = {node: self.original_positions[node] for node in community_nodes_list}
        
        # Create edge traces for this community
        edge_traces = self._create_colored_edge_traces(subgraph, community_positions, config)
        
        # Create node trace for community
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        for node in subgraph.nodes():
            x, y = community_positions[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            
            # Color by degree within community
            degree = subgraph.degree(node)
            node_color.append(degree)
            
            # Size based on degree
            size_range = config.get('node_size_range', (20, 50))
            max_degree = max(dict(subgraph.degree()).values()) if subgraph.degree() else 1
            normalized_size = np.interp(degree, [0, max_degree], size_range)
            node_size.append(normalized_size)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if config.get('show_labels', True) else 'markers',
            hoverinfo='text',
            text=node_text if config.get('show_labels', True) else None,
            textposition="top center",
            textfont=dict(size=config.get('label_size', 10)),
            marker=dict(
                showscale=True,
                colorscale='plasma',
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Degree Centrality',
                    xanchor='left'
                ),
                line_width=2
            ),
            showlegend=False
        )
        
        return edge_traces + [node_trace]
        for comm_id, nodes in community_nodes.items():
            meta_graph.add_node(f"Community_{comm_id}", size=len(nodes))
            community_sizes[comm_id] = len(nodes)

            # Calculate centroid position of community
            positions = []
            for node in nodes:
                if node in self.original_graph.nodes():
                    # Use positions from the original layout
                    if node in self.original_positions:
                        positions.append(self.original_positions[node])

            if positions:
                community_positions[comm_id] = np.mean(positions, axis=0)
            else:
                # Fallback position
                community_positions[comm_id] = (np.random.random(2) - 0.5) * 2

        # Add edges between communities based on inter-community connections
        for u, v in self.original_graph.edges():
            comm_u = self.original_communities.get(u, 0)
            comm_v = self.original_communities.get(v, 0)
            if comm_u != comm_v:
                meta_node_u = f"Community_{comm_u}"
                meta_node_v = f"Community_{comm_v}"

                if meta_graph.has_edge(meta_node_u, meta_node_v):
                    meta_graph[meta_node_u][meta_node_v]['weight'] += 1
                else:
                    meta_graph.add_edge(meta_node_u, meta_node_v, weight=1)

        # Create Plotly figure
        return self._create_interactive_community_figure(meta_graph, community_sizes, community_positions, config)

    def _create_detailed_community_view(self, community_id, config):
        """Create detailed view showing individual nodes within a selected community."""
        # Get nodes in the selected community
        community_nodes = [node for node, comm_id in self.original_communities.items() if comm_id == community_id]

        # Create subgraph for this community
        subgraph = self.original_graph.subgraph(community_nodes).copy()

        # Calculate positions for the subgraph
        positions = {node: self.original_positions[node] for node in community_nodes if node in self.original_positions}

        # Create Plotly figure for detailed view
        return self._create_interactive_detail_figure(subgraph, positions, community_id, config)

    def _create_interactive_community_figure(self, meta_graph, community_sizes, community_positions, config):
        """Create interactive Plotly figure for community-level view with navigation buttons."""

        # Prepare node data
        node_x, node_y, node_text, node_size, node_color, node_ids = [], [], [], [], [], []

        for node in meta_graph.nodes():
            comm_id = int(node.split('_')[1])
            x, y = community_positions[comm_id]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>{community_sizes[comm_id]} units<br><b>Community {comm_id}</b>")
            node_size.append(np.sqrt(community_sizes[comm_id]) * 10 + 20)
            node_color.append(comm_id)
            node_ids.append(comm_id)

        # Prepare edge data
        edge_x, edge_y = [], []

        for edge in meta_graph.edges(data=True):
            node1, node2 = edge[0], edge[1]
            comm1 = int(node1.split('_')[1])
            comm2 = int(node2.split('_')[1])

            x1, y1 = community_positions[comm1]
            x2, y2 = community_positions[comm2]

            edge_x.extend([x1, x2, None])
            edge_y.extend([y1, y2, None])

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create node trace with custom data for navigation
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            customdata=node_ids,  # Store community IDs
            marker=dict(
                showscale=True,
                colorscale=config.get('color_scheme', 'viridis'),
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Community ID',
                    xanchor='left'
                ),
                line_width=2))

        # Create buttons for drill-down to each community
        buttons = []
        community_ids = sorted(community_sizes.keys())

        # Add button to return to overview (always first)
        buttons.append(dict(
            label="Overview",
            method="relayout",
            args=[{
                "title.text": 'Hierarchical Network View - Communities<br><i>Use buttons below to explore individual communities</i>',
                "xaxis.showgrid": False,
                "yaxis.showgrid": False
            }]
        ))

        # Add buttons for each community
        for comm_id in community_ids:
            buttons.append(dict(
                label=f"Community {comm_id} ({community_sizes[comm_id]} units)",
                method="relayout",
                args=[{
                    "title.text": f'Community {comm_id} - Detailed View<br><i>Showing {community_sizes[comm_id]} units and their connections</i>',
                    "xaxis.showgrid": False,
                    "yaxis.showgrid": False
                }]
            ))

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text='Hierarchical Network View - Communities<br><i>Use buttons below to explore individual communities</i>',
                               font=dict(size=16)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=60),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=config.get('figsize', (800, 600))[0],
                           height=config.get('figsize', (800, 600))[1],
                           updatemenus=[dict(
                               type="buttons",
                               direction="left",
                               buttons=buttons,
                               pad={"r": 10, "t": 10},
                               showactive=True,
                               x=0,
                               xanchor="left",
                               y=1.1,
                               yanchor="top"
                           )]
                       ))

        return fig

    def create_drilldown_html(self, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a complete HTML page with drill-down navigation between community views.

        Returns:
            HTML string with interactive navigation
        """
        try:
            if not PLOTLY_AVAILABLE:
                raise ImportError("Plotly is required for drill-down HTML generation")

            # Merge configurations
            viz_config = self.default_config.copy()
            if config:
                viz_config.update(config)

            # Create community-level view
            community_fig = self._create_community_level_view(viz_config)

            # Create detailed views for each community
            community_ids = sorted(set(self.original_communities.values()))
            detail_figs = {}

            for comm_id in community_ids:
                detail_fig = self._create_detailed_community_view(comm_id, viz_config)
                detail_figs[comm_id] = detail_fig

            # Generate HTML with navigation
            html_content = self._generate_drilldown_html(community_fig, detail_figs, viz_config)

            return html_content

        except Exception as e:
            self.logger.error(f"Error creating drill-down HTML: {e}")
            raise

    def _generate_drilldown_html(self, community_fig, detail_figs, config):
        """Generate complete HTML with JavaScript navigation."""

        # Convert figures to HTML divs
        community_div = community_fig.to_html(include_plotlyjs=False, full_html=False)

        detail_divs = {}
        for comm_id, fig in detail_figs.items():
            detail_divs[comm_id] = fig.to_html(include_plotlyjs=False, full_html=False)

        # Create navigation buttons
        community_ids = sorted(detail_figs.keys())
        nav_buttons = []

        for comm_id in community_ids:
            community_size = len([node for node, c_id in self.original_communities.items() if c_id == comm_id])
            nav_buttons.append(f'''
                <button onclick="showCommunity({comm_id})" class="nav-button">
                    Community {comm_id} ({community_size} units)
                </button>
            ''')

        nav_buttons_html = '\n'.join(nav_buttons)

        # Complete HTML template
        html_template = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Hierarchical Network Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .nav-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }}
        .nav-button {{
            padding: 8px 16px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
        }}
        .nav-button:hover {{
            background: #0056b3;
        }}
        .nav-button.active {{
            background: #28a745;
        }}
        .overview-button {{
            background: #6c757d !important;
        }}
        .overview-button:hover {{
            background: #545b62 !important;
        }}
        .view-container {{
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 10px;
            background: white;
        }}
        .hidden {{
            display: none;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .description {{
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hierarchical Network Visualization</h1>
        <p class="description">
            Explore the correlation network at different levels. Start with the community overview,
            then drill down into individual communities to see detailed relationships.
        </p>

        <div class="nav-bar">
            <button onclick="showOverview()" class="nav-button overview-button active" id="overview-btn">
                Overview - All Communities
            </button>
            {nav_buttons_html}
        </div>

        <div class="view-container">
            <!-- Community Overview -->
            <div id="overview-view">
                {community_div}
            </div>

            <!-- Detailed Community Views -->
            {"".join([f'<div id="community-{comm_id}-view" class="hidden">{detail_divs[comm_id]}</div>' for comm_id in community_ids])}
        </div>
    </div>

    <script>
        let currentView = 'overview';

        function showOverview() {{
            // Hide all views
            document.querySelectorAll('[id$="-view"]').forEach(el => el.classList.add('hidden'));

            // Show overview
            document.getElementById('overview-view').classList.remove('hidden');

            // Update button states
            document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
            document.getElementById('overview-btn').classList.add('active');

            currentView = 'overview';
        }}

        function showCommunity(communityId) {{
            // Hide all views
            document.querySelectorAll('[id$="-view"]').forEach(el => el.classList.add('hidden'));

            // Show selected community
            document.getElementById(`community-${{communityId}}-view`).classList.remove('hidden');

            // Update button states
            document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            currentView = `community-${{communityId}}`;
        }}

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            showOverview();
        }});
    </script>
</body>
</html>
        '''

        return html_template

    def _create_interactive_detail_figure(self, subgraph, positions, community_id, config):
        """Create interactive Plotly figure for detailed community view."""

        # Prepare node data
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []

        for node in subgraph.nodes():
            x, y = positions[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))

            # Color by degree centrality within community
            degree = subgraph.degree(node)
            node_color.append(degree)

            # Size based on degree
            size_range = config['node_size_range']
            max_degree = max(dict(subgraph.degree()).values()) if subgraph.degree() else 1
            normalized_size = np.interp(degree, [0, max_degree], size_range)
            node_size.append(normalized_size)

        # Prepare edge data
        edge_x, edge_y = [], []

        for edge in subgraph.edges(data=True):
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text' if config.get('show_labels', True) else 'markers',
            hoverinfo='text',
            text=node_text if config.get('show_labels', True) else None,
            textposition="top center",
            textfont=dict(size=config.get('label_size', 12)),
            marker=dict(
                showscale=True,
                colorscale=config.get('color_scheme', 'viridis'),
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Degree Centrality',
                    xanchor='left'
                ),
                line_width=2))

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text=f'Community {community_id} - Detailed View<br><i>Showing {len(subgraph.nodes())} units and their connections</i>',
                               font=dict(size=16)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=60),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=config.get('figsize', (800, 600))[0],
                           height=config.get('figsize', (800, 600))[1]))

        return fig


# Test function
if __name__ == "__main__":
    print("Testing network visualization module...")

    if not NETWORKX_AVAILABLE:
        print("NetworkX not available, skipping tests")
        exit()

    # Create sample correlation matrix
    np.random.seed(42)
    n_units = 10
    units = [f'Company_{i+1}' for i in range(n_units)]

    # Generate sample correlation matrix with some structure
    corr_matrix = pd.DataFrame(
        np.random.uniform(-0.8, 0.8, (n_units, n_units)),
        index=units,
        columns=units
    )

    # Make it symmetric and with 1s on diagonal
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix.values, 1.0)

    # Add some block structure (communities)
    for i in range(5):
        for j in range(5):
            if i != j:
                corr_matrix.iloc[i, j] = np.random.uniform(0.4, 0.8)
                corr_matrix.iloc[j, i] = corr_matrix.iloc[i, j]

    print("Sample correlation matrix shape:", corr_matrix.shape)

    # Test network creation
    visualizer = NetworkVisualizer()

    try:
        # Create network
        graph = visualizer.create_network_from_correlation(corr_matrix, {'threshold': 0.3})
        print(f"Network created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Detect communities
        communities = visualizer.detect_communities(graph)
        print(f"Communities detected: {len(set(communities.values()))} groups")

        # Calculate positions
        positions = visualizer.calculate_node_positions(graph, 'spring')
        print("Node positions calculated")

        # Network statistics
        stats = visualizer.get_network_statistics(graph)
        print("Network statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}")

        # Test matplotlib visualization
        fig_static = visualizer.create_matplotlib_network(graph, positions, communities)
        print("Static network visualization created")

        # Test Plotly visualization if available
        if PLOTLY_AVAILABLE:
            fig_interactive = visualizer.create_plotly_network(graph, positions, communities)
            print("Interactive network visualization created")
        else:
            print("Plotly not available, skipping interactive visualization")

        print("Network visualization module test completed successfully")

    except Exception as e:
        print(f"Error in testing: {e}")
        import traceback
        traceback.print_exc()