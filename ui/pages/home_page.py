"""首页: 服务器连接 + 模型状态(image/tracker/video 启用状态 + GPU 显存)。"""

from __future__ import annotations

from PySide6.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from ..context import AppContext
from ..workers import run_api

# 模型键 -> 显示名(/model/status 返回的 image/tracker/video)
_MODEL_TITLES = (("image", "图像分割模型"),
                 ("tracker", "追踪模型"),
                 ("video", "视频模型"))


class HomePage(QWidget):
    """连接服务器(以 GET /model/status 作健康检查), 显示模型与 GPU 状态。"""

    DEFAULT_URL = "http://127.0.0.1:8000"

    def __init__(self, context: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = context
        self._connecting = False
        self._setup_ui()
        context.connection_changed.connect(self._on_connection_changed)
        self._on_connection_changed(context.connected)

    # ---------------- UI ----------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("服务器地址:"))
        self.url_edit = QLineEdit(self._ctx.server_url)
        self.url_edit.setPlaceholderText(self.DEFAULT_URL)
        self.connect_btn = QPushButton("连接")
        self.connect_btn.setFixedWidth(90)
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        conn_row.addWidget(self.url_edit, stretch=1)
        conn_row.addWidget(self.connect_btn)
        layout.addLayout(conn_row)

        self.conn_label = QLabel("未连接")
        layout.addWidget(self.conn_label)

        box = QGroupBox("模型状态")
        form = QFormLayout(box)
        self.model_labels = {}
        for key, title in _MODEL_TITLES:
            lbl = QLabel("--")
            self.model_labels[key] = lbl
            form.addRow(f"{title}:", lbl)
        self.gpu_label = QLabel("--")
        form.addRow("GPU 显存:", self.gpu_label)
        layout.addWidget(box)
        layout.addStretch()

    # ---------------- 连接 ----------------
    def _on_connect_clicked(self) -> None:
        if self._ctx.connected:
            self._disconnect()
            return
        url = self.url_edit.text().strip() or self.DEFAULT_URL
        self.url_edit.setText(url)
        self._ctx.set_server(url)  # 重建 client, connected=False
        self._connecting = True
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("连接中...")
        self.conn_label.setText("正在连接...")
        run_api(self._ctx.client.model_status,
                on_ok=self._on_status_ok, on_err=self._on_status_err, parent=self)

    def _disconnect(self) -> None:
        self._ctx.set_connected(False)
        self._clear_status()
        self._ctx.status_message.emit("已断开连接")

    def _on_status_ok(self, status: dict) -> None:
        self._connecting = False
        self.connect_btn.setEnabled(True)
        self._ctx.set_connected(True)
        self._show_status(status)
        self._ctx.status_message.emit(f"已连接: {self._ctx.server_url}")

    def _on_status_err(self, msg: str) -> None:
        self._connecting = False
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("连接")
        self.conn_label.setText("连接失败")
        self._ctx.set_connected(False)
        QMessageBox.warning(self, "连接失败", f"无法连接服务器:\n{msg}")
        self._ctx.status_message.emit(f"连接失败: {msg}")

    # ---------------- 状态显示 ----------------
    def _on_connection_changed(self, ok: bool) -> None:
        if not self._connecting:
            self.connect_btn.setText("断开" if ok else "连接")
            self.conn_label.setText(
                f"已连接 {self._ctx.server_url}" if ok else "未连接")
        if not ok and not self._connecting:
            self._clear_status()

    def _show_status(self, status: dict) -> None:
        for key, lbl in self.model_labels.items():
            info = status.get(key) or {}
            enabled = bool(info.get("enabled"))
            lbl.setText("已启用" if enabled else "未启用")
            lbl.setStyleSheet("color: #3fca3f;" if enabled else "color: #888888;")
        try:
            self.gpu_label.setText(f"{float(status.get('gpu_memory_gb', 0)):.2f} GB")
        except (TypeError, ValueError):
            self.gpu_label.setText("--")

    def _clear_status(self) -> None:
        for lbl in self.model_labels.values():
            lbl.setText("--")
            lbl.setStyleSheet("")
        self.gpu_label.setText("--")
