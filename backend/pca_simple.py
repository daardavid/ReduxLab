#!/usr/bin/env python3
"""
Versión simplificada de pca_gui.py para debuggear el problema
"""

import sys

sys.path.append(".")

print("🚀 Iniciando versión simplificada...")

try:
    # Importaciones mínimas
    import tkinter as tk
    from tkinter import messagebox

    # Crear clase básica
    class PCAAppSimple(tk.Tk):
        def __init__(self):
            print("Inicializando ventana principal...")
            super().__init__()

            self.title("PCA Application")
            self.geometry("800x600")
            self.configure(bg="#f8fafc")

            # Frame principal
            main_frame = tk.Frame(self, bg="#f8fafc")
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)

            # Título
            tk.Label(
                main_frame,
                text="🎯 PCA Analysis Application",
                font=("Segoe UI", 18, "bold"),
                bg="#f8fafc",
                fg="#1e293b",
            ).pack(pady=20)

            # Botón de prueba
            tk.Button(
                main_frame,
                text="✅ Aplicación Funcionando",
                command=self.test_function,
                bg="#10b981",
                fg="white",
                font=("Segoe UI", 12, "bold"),
                padx=20,
                pady=10,
            ).pack(pady=20)

            print("✅ Ventana inicializada correctamente")

        def test_function(self):
            messagebox.showinfo("Éxito", "¡La aplicación simplificada funciona!")

    print("Creando instancia de la aplicación...")
    app = PCAAppSimple()

    print("Iniciando mainloop...")
    app.mainloop()

    print("Aplicación cerrada correctamente.")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
