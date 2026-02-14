"""
Preparar datos / Construir matriz dialog.

Allows building the n×p analysis matrix from multiple Excel sheets (choose one or merge two).
"""

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd


def show_data_prep_dialog(app: Any) -> None:
    """Open the Prepare data / Build matrix dialog. Assigns result to app.get_active_sheet()['df']."""
    _DataPrepDialog(app)


class _DataPrepDialog:
    def __init__(self, app: Any):
        self.app = app
        self.loaded: Dict[str, pd.DataFrame] = {}
        self.file_path: Optional[str] = None

        self.win = tk.Toplevel(app)
        self.win.title("Preparar datos / Construir matriz")
        self.win.transient(app)
        self.win.geometry("520x380")
        f = ttk.Frame(self.win, padding=15)
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Label(f, text="Construir matriz n×p desde archivo Excel (una hoja o combinar dos)", style="primary.TLabel").pack(anchor=tk.W)
        ttk.Button(f, text="Cargar archivo Excel...", style="info.Outline.TButton", command=self._load_file).pack(anchor=tk.W, pady=(8, 4))
        self.lbl_file = ttk.Label(f, text="Ningún archivo cargado", style="secondary.TLabel")
        self.lbl_file.pack(anchor=tk.W)

        mode_f = ttk.LabelFrame(f, text="Modo", padding=8)
        mode_f.pack(fill=tk.X, pady=(12, 8))
        self.mode_var = tk.StringVar(value="one")
        ttk.Radiobutton(mode_f, text="Usar una hoja", variable=self.mode_var, value="one", command=self._on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_f, text="Combinar dos hojas (merge)", variable=self.mode_var, value="merge", command=self._on_mode_change).pack(anchor=tk.W)

        self.one_frame = ttk.Frame(f)
        self.one_frame.pack(fill=tk.X, pady=4)
        ttk.Label(self.one_frame, text="Hoja:").pack(side=tk.LEFT, padx=(0, 8))
        self.sheet_one_var = tk.StringVar()
        self.combo_one = ttk.Combobox(self.one_frame, textvariable=self.sheet_one_var, state="readonly", width=30)
        self.combo_one.pack(side=tk.LEFT)

        self.merge_frame = ttk.Frame(f)
        self.merge_frame.pack(fill=tk.X, pady=4)
        ttk.Label(self.merge_frame, text="Hoja A:").pack(side=tk.LEFT, padx=(0, 4))
        self.sheet_a_var = tk.StringVar()
        self.combo_a = ttk.Combobox(self.merge_frame, textvariable=self.sheet_a_var, state="readonly", width=18)
        self.combo_a.pack(side=tk.LEFT, padx=(0, 12))
        self.combo_a.bind("<<ComboboxSelected>>", lambda e: self._update_merge_columns())
        ttk.Label(self.merge_frame, text="Hoja B:").pack(side=tk.LEFT, padx=(0, 4))
        self.sheet_b_var = tk.StringVar()
        self.combo_b = ttk.Combobox(self.merge_frame, textvariable=self.sheet_b_var, state="readonly", width=18)
        self.combo_b.pack(side=tk.LEFT, padx=(0, 12))
        self.combo_b.bind("<<ComboboxSelected>>", lambda e: self._update_merge_columns())
        ttk.Label(self.merge_frame, text="Clave A:").pack(side=tk.LEFT, padx=(0, 4))
        self.col_a_var = tk.StringVar()
        self.combo_col_a = ttk.Combobox(self.merge_frame, textvariable=self.col_a_var, state="readonly", width=12)
        self.combo_col_a.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(self.merge_frame, text="Clave B:").pack(side=tk.LEFT, padx=(0, 4))
        self.col_b_var = tk.StringVar()
        self.combo_col_b = ttk.Combobox(self.merge_frame, textvariable=self.col_b_var, state="readonly", width=12)
        self.combo_col_b.pack(side=tk.LEFT)

        join_f = ttk.Frame(f)
        join_f.pack(fill=tk.X, pady=4)
        ttk.Label(join_f, text="Tipo de join:").pack(side=tk.LEFT, padx=(0, 8))
        self.how_var = tk.StringVar(value="inner")
        ttk.Combobox(join_f, textvariable=self.how_var, values=["inner", "left"], state="readonly", width=8).pack(side=tk.LEFT)

        self._on_mode_change()

        btn_f = ttk.Frame(f)
        btn_f.pack(fill=tk.X, pady=(16, 0))
        ttk.Button(btn_f, text="Cargar en hoja actual", style="success.TButton", command=self._apply).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_f, text="Cerrar", command=self.win.destroy).pack(side=tk.LEFT)

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleccionar archivo Excel",
            filetypes=[("Excel", "*.xlsx *.xls"), ("Todos", "*.*")]
        )
        if not path:
            return
        try:
            from backend import data_loader_module as dl
            self.loaded = dl.load_excel_file(path) or {}
            self.file_path = path
            names = list(self.loaded.keys())
            self.lbl_file.config(text=f"{Path(path).name} — {len(names)} hoja(s)")
            self.combo_one["values"] = names
            self.combo_a["values"] = names
            self.combo_b["values"] = names
            if names:
                self.sheet_one_var.set(names[0])
                self.sheet_a_var.set(names[0])
                self.sheet_b_var.set(names[1] if len(names) > 1 else names[0])
                self._update_merge_columns()
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar: {e}")

    def _on_mode_change(self) -> None:
        if self.mode_var.get() == "one":
            self.one_frame.pack(fill=tk.X, pady=4)
            self.merge_frame.pack_forget()
        else:
            self.one_frame.pack_forget()
            self.merge_frame.pack(fill=tk.X, pady=4)
            self._update_merge_columns()

    def _update_merge_columns(self) -> None:
        a = self.sheet_a_var.get()
        b = self.sheet_b_var.get()
        if a and a in self.loaded:
            cols = list(self.loaded[a].columns)
            self.combo_col_a["values"] = cols
            if cols and not self.col_a_var.get():
                self.col_a_var.set(cols[0])
        if b and b in self.loaded:
            cols = list(self.loaded[b].columns)
            self.combo_col_b["values"] = cols
            if cols and not self.col_b_var.get():
                self.col_b_var.set(cols[0])

    def _apply(self) -> None:
        sheet = self.app.get_active_sheet() if hasattr(self.app, "get_active_sheet") else None
        if not sheet:
            messagebox.showwarning("Advertencia", "No hay hoja activa.")
            return
        if not self.loaded:
            messagebox.showwarning("Advertencia", "Cargue un archivo Excel primero.")
            return

        try:
            if self.mode_var.get() == "one":
                key = self.sheet_one_var.get()
                if not key or key not in self.loaded:
                    messagebox.showwarning("Advertencia", "Seleccione una hoja.")
                    return
                result = self.loaded[key]
                if not isinstance(result, pd.DataFrame):
                    result = pd.DataFrame(result)
            else:
                key_a = self.sheet_a_var.get()
                key_b = self.sheet_b_var.get()
                col_a = self.col_a_var.get()
                col_b = self.col_b_var.get()
                if not key_a or key_a not in self.loaded or not key_b or key_b not in self.loaded:
                    messagebox.showwarning("Advertencia", "Seleccione ambas hojas.")
                    return
                if not col_a or not col_b:
                    messagebox.showwarning("Advertencia", "Seleccione columnas clave para el merge.")
                    return
                from backend.data_prep import merge_tables
                result = merge_tables(
                    self.loaded[key_a],
                    self.loaded[key_b],
                    on_a=[col_a],
                    on_b=[col_b],
                    how=self.how_var.get() or "inner",
                )
            sheet["df"] = result
            sheet["file_path"] = self.file_path or ""
            name = "Matriz construida" if self.mode_var.get() == "merge" else Path(self.file_path or "").stem
            idx = getattr(self.app, "current_sheet_index", 0)
            if hasattr(self.app, "sheets_notebook") and self.app.sheets_notebook and idx < self.app.sheets_notebook.index("end"):
                self.app.sheets_notebook.tab(idx, text=(name[:20] + "..." if len(name) > 20 else name))
            if hasattr(self.app, "status") and self.app.status:
                self.app.status.config(text=f"Datos cargados: {name}")
            from backend.data_validation import validate_matrix_shape
            warnings_list = validate_matrix_shape(sheet["df"])
            if warnings_list:
                self.app.status.config(text=f"{name} — {warnings_list[0]}")
            messagebox.showinfo("Listo", "Matriz cargada en la hoja actual.")
            self.win.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))
