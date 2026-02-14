"""
Plotting utilities for degree distribution analysis.

This module contains reusable functions for creating degree distribution plots,
eliminating code duplication between basic and advanced plotting methods.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as ttk
from ttkbootstrap.constants import BOTH, X, RIGHT, LEFT
from collections import Counter


def validate_graph_and_networkx(graph, logger=None):
    """
    Validate that a graph is available and NetworkX is installed.

    Args:
        graph: NetworkX graph object to validate
        logger: Optional logger for error reporting

    Returns:
        bool: True if validation passes, False otherwise
    """
    if graph is None:
        messagebox.showwarning(
            "No Network Available",
            "No se ha generado ningún grafo de red.\n"
            "Por favor, ejecute primero un análisis de correlación con visualización de red."
        )
        return False

    try:
        import networkx as nx
    except ImportError:
        messagebox.showerror(
            "NetworkX No Disponible",
            "NetworkX es requerido para el análisis de distribución de grado.\n"
            "Por favor instale con: pip install networkx"
        )
        return False

    return True


def calculate_degrees_and_counts(graph):
    """
    Calculate degree distribution from a NetworkX graph.

    Args:
        graph: NetworkX graph object

    Returns:
        tuple: (degrees_list, counts_list, degrees) where:
            - degrees_list: sorted unique degrees
            - counts_list: corresponding counts
            - degrees: raw degrees list
    """
    degrees = [graph.degree(n) for n in graph.nodes()]

    if not degrees:
        messagebox.showwarning("Grafo Vacío", "El grafo no contiene nodos.")
        return None, None, None

    degree_counts = Counter(degrees)
    sorted_degrees = sorted(degree_counts.items())
    degrees_list = [d[0] for d in sorted_degrees]
    counts_list = [d[1] for d in sorted_degrees]

    return degrees_list, counts_list, degrees


def create_degree_window(title="📊 Distribución de Grado del Nodo", width=800, height=600):
    """
    Create and configure a modal window for degree distribution plots.

    Args:
        title: Window title
        width: Window width
        height: Window height

    Returns:
        tk.Toplevel: Configured window
    """
    degree_window = tk.Toplevel()
    degree_window.title(title)
    degree_window.geometry(f"{width}x{height}")
    degree_window.minsize(width - 100, height - 100)

    # Make it modal
    degree_window.transient(degree_window.master)
    degree_window.grab_set()

    # Center the window
    degree_window.update_idletasks()
    x = (degree_window.winfo_screenwidth() // 2) - (width // 2)
    y = (degree_window.winfo_screenheight() // 2) - (height // 2)
    degree_window.geometry(f"{width}x{height}+{x}+{y}")

    return degree_window


def add_statistics_text(ax, degrees, position=(0.02, 0.98), fontsize=10):
    """
    Add network statistics text to a matplotlib axes.

    Args:
        ax: Matplotlib axes object
        degrees: List of node degrees
        position: Text position as (x, y) in axes coordinates
        fontsize: Font size for the text
    """
    total_nodes = len(degrees)
    mean_degree = sum(degrees) / len(degrees)
    max_degree = max(degrees)
    min_degree = min(degrees)

    stats_text = f'Estadísticas de la Red:\n'
    stats_text += f'• Total de Nodos: {total_nodes}\n'
    stats_text += f'• Grado Promedio: {mean_degree:.2f}\n'
    stats_text += f'• Grado Máximo: {max_degree}\n'
    stats_text += f'• Grado Mínimo: {min_degree}'

    ax.text(position[0], position[1], stats_text, transform=ax.transAxes,
            fontsize=fontsize, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def create_plot_ui(window, fig, save_callback, info_text="💡 Esta gráfica muestra la distribución de conexiones en la red"):
    """
    Create the UI elements for a plot window (canvas, toolbar, buttons).

    Args:
        window: Tkinter window to add UI to
        fig: Matplotlib figure
        save_callback: Function to call when save button is clicked
        info_text: Information text to display

    Returns:
        FigureCanvasTkAgg: The created canvas
    """
    # Create frame for the plot
    plot_frame = ttk.Frame(window)
    plot_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

    # Create canvas and add to window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    # Add toolbar frame
    toolbar_frame = ttk.Frame(window)
    toolbar_frame.pack(fill=X, padx=10, pady=(0, 5))

    # Add buttons
    close_btn = ttk.Button(
        toolbar_frame,
        text="Cerrar",
        command=window.destroy,
        style='primary.TButton'
    )
    close_btn.pack(side=RIGHT, padx=5)

    save_btn = ttk.Button(
        toolbar_frame,
        text="Guardar Gráfico",
        command=lambda: save_plot_callback(fig, save_callback),
        style='secondary.TButton'
    )
    save_btn.pack(side=RIGHT, padx=5)

    # Add info label
    info_label = ttk.Label(
        toolbar_frame,
        text=info_text,
        font=('Helvetica', 9, 'italic')
    )
    info_label.pack(side=LEFT, padx=5)

    return canvas


def save_plot_callback(fig, custom_save_callback=None):
    """
    Handle plot saving with optional custom callback.

    Args:
        fig: Matplotlib figure to save
        custom_save_callback: Optional custom save function
    """
    if custom_save_callback:
        custom_save_callback(fig)
    else:
        save_plot(fig)


def save_plot(fig):
    """
    Save a matplotlib figure to file.

    Args:
        fig: Matplotlib figure to save
    """
    try:
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")],
            title="Guardar Gráfico de Distribución de Grado"
        )
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Éxito", f"Gráfico guardado como:\\n{filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Error al guardar el gráfico:\\n{str(e)}")


def create_bar_plot(ax, degrees_list, counts_list, show_values=True, color='steelblue'):
    """
    Create a bar plot for degree distribution.

    Args:
        ax: Matplotlib axes
        degrees_list: List of degrees
        counts_list: List of counts
        show_values: Whether to show values on bars
        color: Bar color
    """
    bars = ax.bar(degrees_list, counts_list, color=color, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    if show_values and len(degrees_list) <= 20:
        for bar, count in zip(bars, counts_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(count)}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Grado del Nodo (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Nodos', fontsize=12, fontweight='bold')


def create_line_plot(ax, degrees_list, counts_list, show_values=True, color='steelblue'):
    """
    Create a line plot for degree distribution.

    Args:
        ax: Matplotlib axes
        degrees_list: List of degrees
        counts_list: List of counts
        show_values: Whether to show values on points
        color: Line color
    """
    ax.plot(degrees_list, counts_list, marker='o', linewidth=2, markersize=6,
           color=color, markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)

    # Add value labels on points
    if show_values and len(degrees_list) <= 15:
        for x, y in zip(degrees_list, counts_list):
            ax.annotate(f'{int(y)}', (x, y), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=9)

    ax.set_xlabel('Grado del Nodo (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Nodos', fontsize=12, fontweight='bold')


def create_scatter_plot(ax, degrees_list, counts_list, show_values=True, color='steelblue'):
    """
    Create a scatter plot for degree distribution.

    Args:
        ax: Matplotlib axes
        degrees_list: List of degrees
        counts_list: List of counts
        show_values: Whether to show values on points
        color: Point color
    """
    ax.scatter(degrees_list, counts_list, s=80, color=color, alpha=0.7, edgecolors='black')

    # Add value labels
    if show_values and len(degrees_list) <= 15:
        for x, y in zip(degrees_list, counts_list):
            ax.annotate(f'{int(y)}', (x, y), textcoords="offset points",
                       xytext=(5,5), ha='left', fontsize=9)

    ax.set_xlabel('Grado del Nodo (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Nodos', fontsize=12, fontweight='bold')


def create_histogram_plot(ax, degrees, color='steelblue'):
    """
    Create a histogram plot for degree distribution.

    Args:
        ax: Matplotlib axes
        degrees: List of degrees
        color: Histogram color
    """
    ax.hist(degrees, bins=max(10, len(set(degrees))), color=color, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Grado del Nodo (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')


def create_loglog_plot(ax, degrees_list, counts_list, show_values=True, color='steelblue'):
    """
    Create a log-log plot for degree distribution.

    Args:
        ax: Matplotlib axes
        degrees_list: List of degrees
        counts_list: List of counts
        show_values: Whether to show values on points
        color: Plot color
    """
    # Filter out zero values for log scale
    non_zero_degrees = [(d, c) for d, c in zip(degrees_list, counts_list) if d > 0 and c > 0]
    if non_zero_degrees:
        x_vals, y_vals = zip(*non_zero_degrees)
        ax.loglog(x_vals, y_vals, marker='o', linewidth=2, markersize=6,
                 color=color, markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)
        ax.set_xlabel('Grado del Nodo (k) - Log Scale', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Nodos - Log Scale', fontsize=12, fontweight='bold')

        # Add value labels for log-log
        if show_values and len(x_vals) <= 10:
            for x, y in zip(x_vals, y_vals):
                ax.annotate(f'({int(x)},{int(y)})', (x, y), textcoords="offset points",
                           xytext=(5,5), ha='left', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No hay datos válidos para escala log-log',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)


def get_color_map():
    """
    Get the color mapping for different visualization types.

    Returns:
        dict: Color mapping dictionary
    """
    return {
        'steelblue': '#4682B4',
        'viridis': plt.cm.viridis(0.7),
        'plasma': plt.cm.plasma(0.7),
        'red': '#DC143C',
        'green': '#228B22',
        'purple': '#8A2BE2',
        'orange': '#FF8C00'
    }


def get_viz_titles():
    """
    Get titles for different visualization types.

    Returns:
        dict: Title mapping dictionary
    """
    return {
        'bar': 'Distribución de Grado del Nodo\\n(Gráfico de Barras)',
        'line': 'Distribución de Grado del Nodo\\n(Gráfico de Líneas)',
        'scatter': 'Distribución de Grado del Nodo\\n(Diagrama de Dispersión)',
        'histogram': 'Distribución de Grado del Nodo\\n(Histograma)',
        'loglog': 'Distribución de Grado del Nodo\\n(Escala Log-Log)'
    }


def get_viz_descriptions():
    """
    Get descriptions for different visualization types.

    Returns:
        dict: Description mapping dictionary
    """
    return {
        'bar': '📊 Gráfico de barras tradicional',
        'line': '📈 Gráfico de líneas para tendencias',
        'scatter': '🔵 Diagrama de dispersión para outliers',
        'histogram': '📊 Histograma binned para redes grandes',
        'loglog': '📏 Escala log-log para redes scale-free'
    }