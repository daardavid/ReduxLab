"""
PCA Application launcher.

Starts the modern GUI from the frontend package.
Run from project root: python main.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from frontend.pca_gui_modern import PCAApp
from frontend.ui_manager import UIManager
from backend.analysis_manager import AnalysisManager
from backend.file_handler import FileHandler

if __name__ == "__main__":
    ui_manager = UIManager(None)
    analysis_manager = AnalysisManager(None)
    file_handler = FileHandler(None)
    app = PCAApp(ui_manager=ui_manager, analysis_manager=analysis_manager, file_handler=file_handler)
    app.ui_manager.app = app
    app.analysis_manager.app = app
    app.file_handler.app = app
    app.setup_application()
    app.mainloop()
