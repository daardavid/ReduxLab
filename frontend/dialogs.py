# dialogs.py
"""
Dialogs Module for PCA GUI Application.

This module contains dialog creation and management methods
for the PCA application.

Author: David Armando Abreu Rosique
Fecha: 2025
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, colorchooser, filedialog, Toplevel
import pandas as pd
from frontend.ui_components import UIComponents
import logging

logger = logging.getLogger("pca_gui.dialogs")


class DialogManager:
    """Clase para gestionar diálogos de la aplicación PCA."""

    def __init__(self, parent_app):
        """Inicializar con referencia a la app principal."""
        self.parent_app = parent_app
        self.ui = UIComponents(parent_app)

    def show_settings_window(self):
        """Ventana de configuración moderna y mejorada."""
        win = self.ui.create_modern_window("⚙️ Configuración", 450, 550)

        # Título principal
        title_frame = tk.Frame(
            win, bg=getattr(self.parent_app, "bg_primary", "#ffffff")
        )
        title_frame.pack(fill="x", pady=(20, 30))

        title_label = tk.Label(
            title_frame,
            text="🎨 Personalización",
            font=("Segoe UI", 16, "bold"),
            bg=getattr(self.parent_app, "bg_primary", "#ffffff"),
            fg=getattr(self.parent_app, "fg_primary", "#1e293b"),
        )
        title_label.pack()

        # Contenedor principal con scroll si es necesario
        main_frame = tk.Frame(win, bg=getattr(self.parent_app, "bg_primary", "#ffffff"))
        main_frame.pack(fill="both", expand=True, padx=30)

        # Sección Tema
        self.ui._create_settings_section(main_frame, "🌙 Tema", 0)
        theme_var = tk.StringVar(value=getattr(self.parent_app, "theme", "light"))
        theme_frame = tk.Frame(
            main_frame, bg=getattr(self.parent_app, "bg_primary", "#ffffff")
        )
        theme_frame.pack(fill="x", pady=(5, 20))

        light_btn = tk.Radiobutton(
            theme_frame,
            text="☀️ Claro",
            variable=theme_var,
            value="light",
            bg=getattr(self.parent_app, "bg_primary", "#ffffff"),
            fg=getattr(self.parent_app, "fg_primary", "#1e293b"),
            selectcolor=getattr(self.parent_app, "accent_color", "#3b82f6"),
            font=("Segoe UI", 10),
        )
        light_btn.pack(side="left", padx=(20, 30))

        dark_btn = tk.Radiobutton(
            theme_frame,
            text="🌙 Oscuro",
            variable=theme_var,
            value="dark",
            bg=getattr(self.parent_app, "bg_primary", "#ffffff"),
            fg=getattr(self.parent_app, "fg_primary", "#1e293b"),
            selectcolor=getattr(self.parent_app, "accent_color", "#3b82f6"),
            font=("Segoe UI", 10),
        )
        dark_btn.pack(side="left")

        # Sección Idioma
        self.ui._create_settings_section(main_frame, "🌍 Idioma / Language", 20)
        lang_var = tk.StringVar(value=getattr(self.parent_app, "lang", "es"))
        lang_frame = tk.Frame(
            main_frame, bg=getattr(self.parent_app, "bg_primary", "#ffffff")
        )
        lang_frame.pack(fill="x", pady=(5, 20))

        es_btn = tk.Radiobutton(
            lang_frame,
            text="🇪🇸 Español",
            variable=lang_var,
            value="es",
            bg=getattr(self.parent_app, "bg_primary", "#ffffff"),
            fg=getattr(self.parent_app, "fg_primary", "#1e293b"),
            selectcolor=getattr(self.parent_app, "accent_color", "#3b82f6"),
            font=("Segoe UI", 10),
        )
        es_btn.pack(side="left", padx=(20, 30))

        en_btn = tk.Radiobutton(
            lang_frame,
            text="🇺🇸 English",
            variable=lang_var,
            value="en",
            bg=getattr(self.parent_app, "bg_primary", "#ffffff"),
            fg=getattr(self.parent_app, "fg_primary", "#1e293b"),
            selectcolor=getattr(self.parent_app, "accent_color", "#3b82f6"),
            font=("Segoe UI", 10),
        )
        en_btn.pack(side="left")

        # Sección Ventana
        self.ui._create_settings_section(main_frame, "📐 Tamaño de Ventana", 20)
        size_var = tk.StringVar(value=self.parent_app.geometry())
        size_entry = self.ui._create_modern_entry(main_frame, size_var, "Ej: 900x700")

        # Sección Fuente
        self.ui._create_settings_section(main_frame, "🔤 Tipografía", 20)
        font_var = tk.StringVar(
            value=getattr(self.parent_app, "custom_font", "Segoe UI")
        )
        font_entry = self.ui._create_modern_entry(
            main_frame, font_var, "Nombre de la fuente"
        )

        fontsize_var = tk.StringVar(
            value=str(getattr(self.parent_app, "custom_fontsize", 10))
        )
        fontsize_entry = self.ui._create_modern_entry(
            main_frame, fontsize_var, "Tamaño (8-16)", width=15
        )

        # Botones de acción
        buttons_frame = tk.Frame(
            win, bg=getattr(self.parent_app, "bg_primary", "#ffffff")
        )
        buttons_frame.pack(fill="x", pady=(30, 20), padx=30)

        def save_and_close():
            self.parent_app.theme = theme_var.get()
            self.parent_app.custom_font = font_var.get() or "Segoe UI"
            try:
                self.parent_app.custom_fontsize = max(
                    8, min(16, int(fontsize_var.get()))
                )
            except:
                self.parent_app.custom_fontsize = 10
            try:
                self.parent_app.geometry(size_var.get())
            except:
                pass
            new_lang = lang_var.get()
            lang_changed = getattr(self.parent_app, "lang", "es") != new_lang
            self.parent_app.lang = new_lang
            self.parent_app.save_settings()
            self.ui.apply_theme()
            self.ui.apply_font_settings()
            self.ui.apply_matplotlib_style()
            if lang_changed:
                self.parent_app.change_language(self.parent_app.lang)
            self.parent_app.sync_gui_from_cfg()
            win.destroy()

        # Botón guardar con estilo moderno
        save_btn = self.ui.create_modern_button(
            buttons_frame,
            text="💾 Guardar Cambios",
            command=save_and_close,
            style="success",
            width=20,
        )
        save_btn.pack(side="right", padx=(10, 0))

        # Botón cancelar
        cancel_btn = self.ui.create_modern_button(
            buttons_frame,
            text="❌ Cancelar",
            command=win.destroy,
            style="secondary",
            width=15,
        )
        cancel_btn.pack(side="right")

        win.grab_set()
        win.focus_set()

        # Centrar ventana
        win.update_idletasks()
        x = (win.winfo_screenwidth() // 2) - (win.winfo_width() // 2)
        y = (win.winfo_screenheight() // 2) - (win.winfo_height() // 2)
        win.geometry(f"+{x}+{y}")

    def show_about_window(self):
        """Muestra la ventana Acerca de."""
        import webbrowser

        about_text = (
            "# Acerca de nosotros\n\n"
            "Este programa fue desarrollado por David Armando Abreu Rosique.\n\n"
            "Historia: Esta aplicación nació para facilitar el análisis de datos (indicadores y variables) y el uso de técnicas como componentes principales para usuarios no expertos.\n\n"
            "Agradezco a todo el equipo del Instituto de Investigaciones Económicas de la UNAM.\n\n"
            "Contacto: davidabreu1110@gmail.com.\n\n"
            "¿Te gusta el programa? Puedes apoyarme invitándome un café en Ko-fi.\n"
        )
        win = Toplevel(self.parent_app)
        win.title("Acerca de nosotros")
        win.geometry("600x400")
        frame = tk.Frame(win)
        frame.pack(fill="both", expand=True)
        txt = tk.Text(frame, wrap="word", font=("Arial", 12))
        txt.insert("1.0", about_text)
        txt.config(state="disabled")
        txt.pack(side="top", fill="both", expand=True)
        scroll = tk.Scrollbar(frame, command=txt.yview)
        txt.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        btn_kofi = tk.Button(
            frame,
            text="Visitar Ko-fi",
            bg="#ffdd57",
            font=("Arial", 11, "bold"),
            command=lambda: webbrowser.open("https://ko-fi.com/daardavid"),
        )
        btn_kofi.pack(side="bottom", pady=10)

    def show_manual_window(self):
        """Muestra la ventana de manual."""
        manual_text = (
            "# Manual de la aplicación\n\n"
            "Esta aplicación permite realizar análisis PCA sobre datos.\n\n"
            "- **Nuevo Proyecto**: Crea un nuevo proyecto.\n"
            "- **Abrir Proyecto**: Carga un proyecto guardado.\n"
            "- **Guardar Proyecto**: Guarda el estado actual.\n\n"
            "Puedes seleccionar indicadores, unidades y años, y ejecutar análisis de serie de tiempo, corte transversal o panel.\n\n"
            "**Ejemplo de uso:**\n\n1. Crea un nuevo proyecto.\n2. Selecciona el archivo de datos.\n3. Elige los indicadores, unidades y años.\n4. Ejecuta el análisis.\n\nPara más detalles, consulta el manual completo."
        )
        win = Toplevel(self.parent_app)
        win.title("Manual")
        win.geometry("600x500")
        frame = tk.Frame(win)
        frame.pack(fill="both", expand=True)
        txt = tk.Text(frame, wrap="word", font=("Arial", 12))
        txt.insert("1.0", manual_text)
        txt.config(state="disabled")
        txt.pack(side="left", fill="both", expand=True)
        scroll = tk.Scrollbar(frame, command=txt.yview)
        txt.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

    def gui_select_imputation_strategy(self):
        """Diálogo para seleccionar estrategia de imputación."""
        estrategia = None
        params = {}

        STRATEGIAS = [
            (
                "interpolacion",
                "Interpolación (lineal por defecto, o especificar método)",
            ),
            ("mean", "Rellenar con la Media"),
            ("median", "Rellenar con la Mediana"),
            ("most_frequent", "Rellenar con el Valor Más Frecuente (moda)"),
            ("ffill", "Rellenar con valor anterior (Forward Fill)"),
            ("bfill", "Rellenar con valor siguiente (Backward Fill)"),
            ("iterative", "Imputación Iterativa (multivariada)"),
            ("knn", "Imputación KNN (basada en vecinos)"),
            ("valor_constante", "Rellenar con un Valor Constante específico"),
            ("eliminar_filas", "Eliminar filas con datos faltantes"),
            ("ninguna", "No aplicar ninguna imputación (mantener NaNs)"),
        ]

        win = Toplevel(self.parent_app)
        win.title("Selecciona Estrategia de Imputación")
        win.geometry("480x420")
        tk.Label(
            win,
            text="Selecciona cómo quieres imputar los datos faltantes:",
            font=("Arial", 11, "bold"),
        ).pack(pady=10)

        estrategia_var = tk.StringVar(value="interpolacion")
        for key, txt in STRATEGIAS:
            tk.Radiobutton(
                win,
                text=txt,
                variable=estrategia_var,
                value=key,
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=25)

        valor_entry = tk.Entry(win)

        def on_radio_change(*a):
            if estrategia_var.get() == "valor_constante":
                valor_entry.pack(pady=8)
                valor_entry.delete(0, tk.END)
                valor_entry.insert(0, "0")
            else:
                valor_entry.pack_forget()

        estrategia_var.trace_add("write", on_radio_change)

        def on_ok():
            nonlocal estrategia, params
            estrategia = estrategia_var.get()
            if estrategia == "valor_constante":
                try:
                    params["valor_constante"] = float(valor_entry.get())
                except Exception:
                    params["valor_constante"] = valor_entry.get()
            win.destroy()

        tk.Button(win, text="OK", command=on_ok, bg="lightblue").pack(pady=14)
        win.transient(self.parent_app)
        win.grab_set()
        self.parent_app.wait_window(win)
        return estrategia, params

    def gui_select_n_components(
        self, max_components, suggested_n_90=None, suggested_n_95=None
    ):
        """
        Abre un diálogo para que el usuario seleccione el número de componentes principales a retener.
        """
        selected_n = [
            max_components
        ]  # Usamos lista para tener referencia mutable en closure

        win = Toplevel(self.parent_app)
        win.title("Seleccionar número de componentes principales")
        win.geometry("420x250")
        mensaje = f"Ingrese cuántos componentes principales deseas retener (1-{max_components}).\n"
        if suggested_n_90:
            mensaje += f"Sugerencia: {suggested_n_90} componentes ≈ 80% varianza.\n"
        if suggested_n_95:
            mensaje += f"Sugerencia: {suggested_n_95} componentes ≈ 90% varianza.\n"
        mensaje += "Deja vacío para usar todos."

        tk.Label(win, text=mensaje, justify="left", wraplength=400).pack(pady=16)

        entry = tk.Entry(win)
        entry.pack(pady=6)
        entry.focus_set()

        def on_ok():
            value = entry.get().strip()
            if not value:
                selected_n[0] = max_components
            else:
                try:
                    n = int(value)
                    if 1 <= n <= max_components:
                        selected_n[0] = n
                    else:
                        messagebox.showerror(
                            "Error", f"El número debe estar entre 1 y {max_components}."
                        )
                        return
                except Exception:
                    messagebox.showerror(
                        "Error", "Debes ingresar un número entero válido."
                    )
                    return
            win.destroy()

        tk.Button(win, text="OK", command=on_ok, bg="lightblue", width=12).pack(pady=16)
        win.grab_set()
        self.parent_app.wait_window(win)
        return selected_n[0]
