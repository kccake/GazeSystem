"""应用级共享上下文(客户端 + 连接状态), 各页面通过它访问后端。"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from GazeSystem_v1.api.client import Sam3Client


class AppContext(QObject):
    """持有 Sam3Client 与连接状态, 状态变化经 signal 通知外壳/页面。"""

    connection_changed = Signal(bool)
    status_message = Signal(str)  # 供状态栏显示

    def __init__(self, server_url: str = "http://127.0.0.1:8000", parent=None):
        super().__init__(parent)
        self._server_url = server_url
        self.client: Sam3Client = Sam3Client(server_url)
        self.connected: bool = False

    @property
    def server_url(self) -> str:
        return self._server_url

    def set_server(self, url: str) -> None:
        """切换服务器地址: 重建 Sam3Client, 连接状态复位。"""
        self._server_url = url
        self.client = Sam3Client(url)
        self.set_connected(False)

    def set_connected(self, ok: bool) -> None:
        """更新连接状态并发 connection_changed。"""
        if self.connected != ok:
            self.connected = ok
            self.connection_changed.emit(ok)
