"""应用入口: python -m GazeSystem_v1.ui.app"""

from __future__ import annotations

import sys
from typing import Tuple

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from .context import AppContext
from .main_window import MainWindow


def _dark_palette() -> QPalette:
    """深色调色板: 背景 #1e1e1e / 面板 #2d2d2d / 文字 #e0e0e0 / 高亮 #3daee9。"""
    p = QPalette()
    p.setColor(QPalette.Window, QColor("#1e1e1e"))
    p.setColor(QPalette.WindowText, QColor("#e0e0e0"))
    p.setColor(QPalette.Base, QColor("#2d2d2d"))
    p.setColor(QPalette.AlternateBase, QColor("#252526"))
    p.setColor(QPalette.Text, QColor("#e0e0e0"))
    p.setColor(QPalette.Button, QColor("#2d2d2d"))
    p.setColor(QPalette.ButtonText, QColor("#e0e0e0"))
    p.setColor(QPalette.Highlight, QColor("#3daee9"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.Link, QColor("#3daee9"))
    p.setColor(QPalette.ToolTipBase, QColor("#2d2d2d"))
    p.setColor(QPalette.ToolTipText, QColor("#e0e0e0"))
    p.setColor(QPalette.PlaceholderText, QColor("#808080"))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor("#666666"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#666666"))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#666666"))
    return p


def create_app() -> Tuple[QApplication, MainWindow]:
    """创建 QApplication(Fusion 深色) + AppContext + 主窗口。"""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    context = AppContext()
    window = MainWindow(context)
    return app, window


def main() -> int:
    app, window = create_app()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
