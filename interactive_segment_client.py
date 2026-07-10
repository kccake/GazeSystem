"""交互式点击分割：API 客户端、后台工作线程、页面。"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
) # 搞了一堆需要的控件, 这个还不知道, 到时候再说

from .widgets import SegmentationLabel

class SegmentionApiClient:
    """调用后端 /interactive/* 接口。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    def set_base_url(self, url:str) -> None:
        self.base_url = url.strip().rstrip("/") # 为了方便重新设置url, 不用销毁Client, 就能重置url


    # 将后端暴露出来的接口在前端写清楚怎么调用
    def _post(self, path: str, data: dict) -> dict:
        url = self.base_url + path
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"}, # 按照JSON格式来解析body
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8")) # 我的图片也被encode成了base64, 所以也可以用json读
        
    def start_session(self, image_path: Path | str) -> str:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8") # 将图片用base64编码
        r = self._post("/interactive/start", {"image_base64": b64})
        if not r.get("success"):
            raise RuntimeError(r.get("message", "启动会话失败")) # 如果success字段为False
        self.session_id = r["session_id"]
        return self.session_id
    
    def add_point(self, x: float, y: float, label: int) ->dict:
        # 由于在后端已经写好了异常的报错, 所以在前端只需要盯一下会话是否启动就可以了
        if self.session_id is None:
            raise RuntimeError("会话未启动")
        return self._post(
           "/interactive/add_point",
            {
                "session_id": self.session_id,
                "point": [float(x), float(y)],
                "label": int(label),
            }, 
        )
    
    def pause(self) -> dict:
        if self.session_id is None:
            raise RuntimeError("会话未启动")
        return self._post("/interactive/pause", {"session_id": self.session_id})
    
    def resume(self) -> dict:
        if self.session_id is None:
            raise RuntimeError("会话未启动")
        return self._post("/interactive/resume", {"session_id": self.session_id})
    
    def clear(self) -> dict:
        if self.session_id is None:
            raise RuntimeError("会话未启动")
        return self._post("/interactive/clear", {"session_id": self.session_id})
    
    def save(self, return_format: str = "mask") -> dict:
        if self.session_id is None:
            raise RuntimeError("会话未启动")
        return self._post(
            "/interactive/save",
            {"session_id": self.session_id, "return_format": return_format},
        )

    def close(self) -> None:
        if self.session_id is None:
            return
        try:
            url = f"{self.base_url}/interactive/session/{self.session_id}"
            req = urllib.request.Request(url, method="DELETE")
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception:
            pass
        finally:
            self.session_id = None
        
class SegmentationWorker(QObject):
    """在子线程里跑 HTTP 请求，避免卡住 UI。"""

    # 先申请一堆表达状态的信号
    started = Signal(str)                      # session_id
    point_result = Signal(object, int, str)    # mask_np, num_points, message
    paused = Signal(str)
    resumed = Signal(str)
    cleared = Signal(str)
    saved = Signal(object, object, str, str)   # mask, shape, mask_base64, message
    error = Signal(str)
    status = Signal(str)

    def __init__(self, base_url: str = "http://127.0.0.1:8000", parent=None):
        super().__init__(parent)
        self.client = SegmentionApiClient(base_url)
    
    @Slot(str)
    def set_base_url(self, url: str) -> None:
        self.client.set_base_url(url)
    
    @Slot(str)
    def start_session(self, image_path: str) -> None:
        self.status.emit("正在连接后端启动会话...")
        try:
            sid = self.client.start_session(image_path)
            self.started.emit(sid)
        except Exception as e:
            self.error.emit(f"启动会话失败: {e}") # 发射启动失败的信号
    
    @Slot(float, float, int) # Slot里的参数类型负责告诉Qt这个槽函数接收什么样的类型函数, 与函数定义相对应
    def add_point(self, x: float, y: float, label: int) -> None:
        try:
            r = self.client.add_point(x, y, label)
            if not r.get("success"):
                self.error.emit(r.get("message", "分割失败"))
                return
            mask = np.array(r["mask"], dtype=np.uint8) if r.get("mask") is not None else None
            self.point_result.emit(mask, r.get("num_points", 0), r.get("message", ""))
        except Exception as e:
            self.error.emit(f"添加点失败: {e}")
    
    @Slot()
    def pause(self) -> None:
        try:
            r = self.client.pause()
            self.paused.emit(r.get("message", "已暂停"))
        except Exception as e:
            self.error.emit(f"暂停失败: {e}")

    @Slot()
    def resume(self) -> None:
        try:
            r = self.client.resume()
            self.resumed.emit(r.get("message", "已恢复"))
        except Exception as e:
            self.error.emit(f"恢复失败: {e}")
    
    @Slot()
    def clear(self) -> None:
        try:
            r = self.client.clear()
            self.cleared.emit(r.get("message", "已清除"))
        except Exception as e:
            self.error.emit(f"清除失败: {e}")

    @Slot(str)
    def save(self, return_format: str = "mask") -> None:
        try:
            r = self.client.save(return_format)
            if not r.get("success"):
                self.error.emit(r.get("message", "保存失败"))
                return
            mask = np.array(r["mask"], dtype=np.uint8) if r.get("mask") is not None else None
            self.saved.emit(mask, r.get("shape"), r.get("mask_base64"), r.get("message", ""))
        except Exception as e:
            self.error.emit(f"保存失败: {e}")
    
    @Slot()
    def close_session(self) -> None:
        self.client.close()

class InteractiveSegmentationPage(QWidget):
    """交互点击分割页面。"""

    image_loaded = Signal(Path)
    point_added = Signal(float, float, int)
    clear_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    save_requested = Signal()
    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_path: Optional[Path] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self) # 垂直布局
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QWidget()
        toolbar.setFixedHeight(44)
        toolbar.setStyleSheet(
            """
            QWidget {
                background-color: #252526;
                border-bottom: 1px solid #3c3c3c;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:pressed { background-color: #606060; }
            QRadioButton { color: #cccccc; font-size: 13px; }
            QLineEdit {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #505050;
                border-radius: 4px;
                padding: 2px 6px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #3c3c3c;
            }
            QSlider::handle:horizontal {
                background: #888888;
                width: 12px;
                margin: -4px 0;
            }
            """
        )
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(8)

        self.back_btn = QPushButton("返回图片")
        self.open_btn = QPushButton("打开图片")

        self.positive_btn = QRadioButton("正样本")
        self.negative_btn = QRadioButton("负样本")
        self.positive_btn.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.positive_btn, 1)
        self.mode_group.addButton(self.negative_btn, 0)
        self.mode_group.idClicked.connect(self._on_mode_changed)

        self.clear_btn = QPushButton(" 清除点")
        self.pause_btn = QPushButton(" 暂停")
        self.save_btn = QPushButton("保存结果")

        self.alpha_slider = QSlider()
        self.alpha_slider.setOrientation(Qt.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(128)
        self.alpha_slider.setFixedWidth(100)

        self.server_input = QLineEdit()
        self.server_input.setPlaceholderText("服务地址")
        self.server_input.setFixedWidth(180)
        self.server_input.setText("http://127.0.0.1:8000") 

        tb_layout.addWidget(self.back_btn)
        tb_layout.addSpacing(10)
        tb_layout.addWidget(self.open_btn)
        tb_layout.addSpacing(10)
        tb_layout.addWidget(self.positive_btn)
        tb_layout.addWidget(self.negative_btn)
        tb_layout.addWidget(self.clear_btn)
        tb_layout.addWidget(self.pause_btn)
        tb_layout.addWidget(self.save_btn)
        tb_layout.addSpacing(10)
        tb_layout.addWidget(QLabel("透明度:"))
        tb_layout.addWidget(self.alpha_slider)
        tb_layout.addStretch()
        tb_layout.addWidget(QLabel("服务:"))
        tb_layout.addWidget(self.server_input)

        layout.addWidget(toolbar)

        self.label = SegmentationLabel()
        layout.addWidget(self.label, stretch=1)

        self.status_label = QLabel("请打开或选择一张图片开始交互分割")
        self.status_label.setStyleSheet(
            "background-color: #1a1a1a; color: #888888; padding: 4px 8px;"
        )
        self.status_label.setFixedHeight(26)
        layout.addWidget(self.status_label)

        self.back_btn.clicked.connect(self.back_requested.emit)
        self.open_btn.clicked.connect(self._choose_image)
        self.clear_btn.clicked.connect(self.clear_requested.emit)
        self.pause_btn.clicked.connect(self._on_pause_clicked)
        self.save_btn.clicked.connect(self.save_requested.emit)
        self.alpha_slider.valueChanged.connect(self.label.set_mask_alpha)
        self.label.point_added.connect(self.point_added.emit)

    def set_server_url(self, url: str) -> None:
        self.server_input.setText(url)

    def get_server_url(self) -> str:
        return self.server_input.text().strip()
        
    def load_image(self, path: Path) -> bool:
        pixmap = QPixmap(str(path)) # 加载图片
        if pixmap.isNull():
            QMessageBox.warning(self, "提示", f"无法加载图片: {path}")
            return False
        self._image_path = path
        self.label.setEnabled(True)
        self.label.set_image(pixmap)
        self.set_paused_state(False)
        self.set_status(f"已加载: {path.name}，点击目标区域开始分割") # 这怎么没定义呢
        self.image_loaded.emit(path)
        return True

    # 搞了一堆意义不是很明确的函数 **************
    def get_image_path(self) -> Optional[Path]:
        return self._image_path
    
    def set_mask(self, mask) -> None:
        self.label.set_mask(mask)

    def clear_local(self) -> None:
        self.label.clear_points()
        self.set_status("已清除点击点")
    
    def set_paused_state(self, paused: bool) -> None:
        self.pause_btn.setText("继续" if paused else "暂停")
        self.label.setEnabled(not paused)
    
    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
    
    def _on_mode_changed(self, mode_id: int) -> None:
        self.label.set_current_label(mode_id)
    
    def _on_pause_clicked(self) -> None:
        if self.pause_btn.text().startswith("⏸"):
            self.pause_requested.emit()
        else:
            self.resume_requested.emit()
    
    def _choose_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(Path.home()),
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tiff);;所有文件 (*.*)",
        )
        if file_path:
            self.load_image(Path(file_path))
    # *******************************************

