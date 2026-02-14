"""
UI Components for PCA Application

Additional UI components and utilities.
"""

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from frontend.base_analysis_frame import BaseAnalysisFrame


class UIComponents:
    """UI Components class for the PCA application."""

    def __init__(self, parent_app):
        """Initialize with parent app reference."""
        self.parent_app = parent_app

    def create_modern_window(self, title, width=400, height=500, resizable=True):
        """Crea una ventana moderna con el tema aplicado."""
        win = tk.Toplevel(self.parent_app)
        win.title(title)
        win.geometry(f"{width}x{height}")
        win.resizable(resizable, resizable)

        # Usar colores seguros por defecto
        bg_color = getattr(self.parent_app, "bg_primary", "#ffffff")
        win.configure(bg=bg_color)

        # Centrar ventana
        win.update_idletasks()
        x = (win.winfo_screenwidth() // 2) - (width // 2)
        y = (win.winfo_screenheight() // 2) - (height // 2)
        win.geometry(f"{width}x{height}+{x}+{y}")

        return win

    def _create_settings_section(self, parent, title, pady_top=0):
        """Crea una sección con título en la ventana de configuración."""
        section_label = tk.Label(
            parent,
            text=title,
            font=("Segoe UI", 12, "bold"),
            bg=getattr(self.parent_app, "bg_primary", "#ffffff"),
            fg=getattr(self.parent_app, "accent_color", "#3b82f6"),
        )
        section_label.pack(anchor="w", pady=(pady_top, 5))
        return section_label

    def _create_modern_entry(self, parent, textvariable, placeholder="", width=25):
        """Crea un Entry moderno con placeholder."""
        entry_frame = tk.Frame(parent, bg=getattr(self.parent_app, "bg_primary", "#ffffff"))
        entry_frame.pack(fill="x", pady=(5, 10))

        entry = tk.Entry(
            entry_frame,
            textvariable=textvariable,
            font=("Segoe UI", 10),
            bg=getattr(self.parent_app, "bg_secondary", "#f8fafc"),
            fg=getattr(self.parent_app, "fg_primary", "#1e293b"),
            relief="flat",
            bd=1,
            width=width,
            insertbackground=getattr(self.parent_app, "fg_primary", "#1e293b"),
        )
        entry.pack(padx=(20, 0), anchor="w")

        # Añadir placeholder como etiqueta si está vacío
        if placeholder:
            placeholder_label = tk.Label(
                entry_frame,
                text=f"💡 {placeholder}",
                font=("Segoe UI", 8),
                fg=getattr(self.parent_app, "fg_secondary", "#64748b"),
                bg=getattr(self.parent_app, "bg_primary", "#ffffff"),
            )
            placeholder_label.pack(padx=(25, 0), anchor="w")

    def create_modern_button(
        self, parent, text, command=None, style="primary", width=None, height=2
    ):
        """Crea un botón moderno con efectos hover y colores mejorados."""
        # Asegurar que las fuentes estén definidas
        if not hasattr(self.parent_app, "font_button"):
            font = getattr(self.parent_app, "custom_font", "Segoe UI")
            fontsize = getattr(self.parent_app, "custom_fontsize", 10)
            self.parent_app.font_button = (font, fontsize, "normal")

        # Configurar colores según el estilo
        if style == "primary":
            bg_normal = getattr(self.parent_app, "btn_primary", "#3b82f6")
            bg_hover = getattr(self.parent_app, "btn_hover", "#2563eb")
            fg_color = "#ffffff"
        elif style == "success":
            bg_normal = getattr(self.parent_app, "btn_success", "#10b981")
            bg_hover = "#059669"
            fg_color = "#ffffff"
        elif style == "secondary":
            bg_normal = getattr(self.parent_app, "btn_secondary", "#64748b")
            bg_hover = "#475569"
            fg_color = "#ffffff"
        else:
            bg_normal = getattr(self.parent_app, "btn_primary", "#3b82f6")
            bg_hover = getattr(self.parent_app, "btn_hover", "#2563eb")
            fg_color = "#ffffff"

        # Crear botón con configuración moderna
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_normal,
            fg=fg_color,
            font=self.parent_app.font_button,
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            width=width,
            height=height,
        )

        # Añadir efectos hover
        btn.bind("<Enter>", lambda e: btn.configure(bg=bg_hover))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg_normal))

        return btn

    def apply_theme(self):
        """Aplica tema visual moderno con colores mejorados y efectos."""
        # Colores modernos mejorados
        if getattr(self.parent_app, "theme", "light") == "dark":
            # Tema oscuro moderno
            self.parent_app.bg_primary = "#1e1e2e"
            self.parent_app.bg_secondary = "#313244"
            self.parent_app.fg_primary = "#cdd6f4"
            self.parent_app.fg_secondary = "#a6adc8"
            self.parent_app.accent_color = "#89b4fa"
            self.parent_app.success_color = "#a6e3a1"
            self.parent_app.warning_color = "#f9e2af"
            self.parent_app.error_color = "#f38ba8"
            self.parent_app.btn_primary = "#89b4fa"
            self.parent_app.btn_secondary = "#585b70"
            self.parent_app.btn_success = "#a6e3a1"
            self.parent_app.btn_hover = "#74c7ec"
        else:
            # Tema claro moderno
            self.parent_app.bg_primary = "#ffffff"
            self.parent_app.bg_secondary = "#f8fafc"
            self.parent_app.fg_primary = "#1e293b"
            self.parent_app.fg_secondary = "#475569"
            self.parent_app.accent_color = "#3b82f6"
            self.parent_app.success_color = "#10b981"
            self.parent_app.warning_color = "#ef4444"
            self.parent_app.btn_primary = "#3b82f6"
            self.parent_app.btn_secondary = "#64748b"
            self.parent_app.btn_success = "#10b981"
            self.parent_app.btn_hover = "#2563eb"

        # Aplicar colores base
        self.parent_app.configure(bg=self.parent_app.bg_primary)

        # Actualizar widgets recursivamente
        self._update_widget_theme(self.parent_app)

        # Menú con colores modernos
        if hasattr(self.parent_app, "menu_bar"):
            self.parent_app.menu_bar.configure(
                bg=self.parent_app.bg_secondary,
                fg=self.parent_app.fg_primary,
                activebackground=self.parent_app.accent_color,
                activeforeground=self.parent_app.bg_primary,
            )

    def _update_widget_theme(self, parent):
        """Actualiza recursivamente el tema de todos los widgets."""
        for widget in parent.winfo_children():
            widget_class = widget.winfo_class()

            if widget_class == "Label":
                widget.configure(bg=self.parent_app.bg_primary, fg=self.parent_app.fg_primary)
            elif widget_class == "Frame":
                widget.configure(bg=self.parent_app.bg_primary)
                self._update_widget_theme(widget)  # Recursivo para frames
            elif widget_class == "Button":
                # No aplicar tema automático a botones, se manejan individualmente
                pass
            elif widget_class == "Entry":
                widget.configure(
                    bg=self.parent_app.bg_secondary,
                    fg=self.parent_app.fg_primary,
                    insertbackground=self.parent_app.fg_primary,
                    relief="flat",
                    bd=1,
                )
            elif widget_class == "Listbox":
                widget.configure(
                    bg=self.parent_app.bg_secondary,
                    fg=self.parent_app.fg_primary,
                    selectbackground=self.parent_app.accent_color,
                    relief="flat",
                    bd=1,
                )

    def apply_font_settings(self):
        """Aplica configuración de fuente moderna."""
        font = getattr(self.parent_app, "custom_font", "Segoe UI")
        fontsize = getattr(self.parent_app, "custom_fontsize", 10)

        # Fuentes diferenciadas para jerarquía visual
        self.parent_app.font_title = (font, fontsize + 4, "bold")
        self.parent_app.font_button = (font, fontsize, "normal")
        self.parent_app.font_label = (font, fontsize, "normal")
        self.parent_app.font_small = (font, fontsize - 1, "normal")

        # Aplicar a widgets principales
        for widget in self.parent_app.winfo_children():
            if isinstance(widget, tk.Label):
                if hasattr(widget, "_is_title"):
                    widget.configure(font=self.parent_app.font_title)
                else:
                    widget.configure(font=self.parent_app.font_label)

    def apply_matplotlib_style(self):
        import matplotlib.pyplot as plt

        if getattr(self.parent_app, "theme", "light") == "dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")


class ScrollableFrame(ttk.Frame):
    """
    A scrollable frame widget for tkinter using canvas and scrollbars.

    This provides a scrollable container similar to QScrollArea in Qt.
    """

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        # Create canvas and scrollbars
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg='white')
        self.v_scrollbar = ttk.Scrollbar(self, orient=VERTICAL, command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self, orient=HORIZONTAL, command=self.canvas.xview)

        # Configure canvas
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set)

        # Create frame inside canvas
        self.scrollable_frame = ttk.Frame(self.canvas, style='TFrame')

        # Bind frame to canvas
        def update_scrollregion(event=None):
            bbox = self.canvas.bbox("all")
            if bbox:
                self.canvas.configure(scrollregion=bbox)

        self.scrollable_frame.bind("<Configure>", update_scrollregion)

        # Also update immediately
        self.canvas.after(100, update_scrollregion)

        # Create window in canvas
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Pack components
        self.canvas.pack(side=LEFT, fill=BOTH, expand=YES)
        self.v_scrollbar.pack(side=RIGHT, fill=Y)
        self.h_scrollbar.pack(side=BOTTOM, fill=X)

        # Bind mousewheel to canvas
        self._bind_mousewheel()

    def _bind_mousewheel(self):
        """Bind mousewheel to canvas scrolling."""
        def _on_mousewheel(event):
            if not self.canvas.winfo_exists():
                return
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _on_mousewheel_linux(event):
            if not self.canvas.winfo_exists():
                return
            self.canvas.yview_scroll(-1*(event.num-4), "units")

        # Store references for cleanup
        self._mousewheel_handler = _on_mousewheel
        self._mousewheel_linux_handler = _on_mousewheel_linux

        # Windows and MacOS
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Linux
        self.canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", _on_mousewheel_linux)

    def destroy(self):
        """Clean up global bindings before destroying the widget."""
        try:
            if self.canvas.winfo_exists():
                self.canvas.unbind_all("<MouseWheel>")
                self.canvas.unbind_all("<Button-4>")
                self.canvas.unbind_all("<Button-5>")
        except Exception:
            pass
        super().destroy()

    def get_frame(self):
        """Get the scrollable frame to add widgets to."""
        return self.scrollable_frame


class ResponsiveAnalysisFrame(BaseAnalysisFrame):
    """
    Base class for analysis frames with responsive, scrollable content.

    This ensures that analysis frames work well on screens of any size.
    Inherits from BaseAnalysisFrame to get common functionality.
    """

    def __init__(self, parent, app):
        # Initialize BaseAnalysisFrame (which handles ttk.Frame initialization)
        BaseAnalysisFrame.__init__(self, parent, app)

        # Create scrollable container
        self.scrollable_container = ScrollableFrame(self)
        self.scrollable_container.pack(fill=BOTH, expand=YES)

        # Get the frame to add content to
        self.content_frame = self.scrollable_container.get_frame()

    def sync_groups_format(self):
        """Synchronize groups from UniversalGroupManager to legacy format."""
        if not hasattr(self, 'groups'):
            self.groups = {}
        if not hasattr(self, 'group_colors'):
            self.group_colors = {}

        self.groups.clear()  # Reset legacy format
        self.group_colors.clear()  # Reset colors

        if hasattr(self.app, 'group_manager') and self.app.group_manager:
            universal_groups = self.app.group_manager.get_all_groups()

            # Convert from {group_name: {units: [list], color: color}}
            # to {unit: group_name} and {group_name: color}
            for group_name, group_data in universal_groups.items():
                units = group_data.get('units', [])
                color = group_data.get('color', '#FF6B6B')

                # Map each unit to its group
                for unit in units:
                    self.groups[unit] = group_name

                # Store group color
                self.group_colors[group_name] = color

    def create_config_card(self, title, content_callback, fill=X):
        """Create a configuration card within the scrollable area."""
        card = ttk.LabelFrame(self.content_frame, text=f"⚙️ {title}", padding=20)
        card.pack(fill=fill, pady=(0, 20))

        content_frame = ttk.Frame(card)
        content_frame.pack(fill=fill)

        content_callback(content_frame)
        return card
