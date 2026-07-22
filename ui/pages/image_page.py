"""图像分割页: 打开图片建会话, 点/框提示标注 + 自动预测, 组管理, 笔刷编辑, mask/prompt 保存与导入。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFileDialog, QFrame,
                               QHBoxLayout, QInputDialog, QLabel, QMessageBox,
                               QPushButton, QRadioButton, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from ..context import AppContext
from ..prompt_file import load_file, save_image_file
from ..utils import pil_to_qimage, qimage_to_pil
from ..widgets.canvas import AnnotationCanvas
from ..widgets.group_panel import GroupPanel
from ..workers import run_api

# 保存对话框默认目录(GazeSystem_v1/masks, 不存在则回退用户主目录)
_MASKS_DIR = Path(__file__).resolve().parents[2] / "masks"


class ImagePage(QWidget):
    """图像分割页(MainWindow 依赖: 类名 / __init__(context, parent) / shutdown())。

    本地提示状态 self._prompts(gid -> {"points": [(x,y,label)], "box": tuple|None})
    是画布提示显示与 prompt 导出的唯一依据; mask 显示以服务端 predict/delete_group
    返回为准, 笔刷编辑只改本地 mask(仅影响导出, 不回传模型)。
    """

    def __init__(self, context: AppContext, parent=None):
        super().__init__(parent)
        self._ctx = context
        self._sid: Optional[str] = None              # 当前图像会话 id
        self._pil_image: Optional[Image.Image] = None  # 原图(保存叠加图用)
        self._prompts: Dict[int, dict] = {}          # gid -> {"points": [...], "box": ...}
        self._masks: Dict[int, QImage] = {}          # gid -> 灰度 mask QImage
        self._creating = False                       # 会话创建中(禁用相关按钮)
        self._brush_gid: Optional[int] = None        # 笔刷编辑目标组
        self._last_error = ""                        # 状态栏"最后错误"
        self._setup_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._update_buttons()
        self._update_status()

    # ================= UI 搭建 =================
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(self._build_toolbar1())
        layout.addLayout(self._build_toolbar2())

        content = QHBoxLayout()
        self.panel = GroupPanel()
        self.panel.setFixedWidth(220)
        content.addWidget(self.panel)
        self.canvas = AnnotationCanvas()
        content.addWidget(self.canvas, stretch=1)
        layout.addLayout(content, stretch=1)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self.status_label)

    def _build_toolbar1(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self.open_btn = QPushButton("打开图片")
        row.addWidget(self.open_btn)
        row.addWidget(_vline())

        # 标注模式(点/框, 单选)
        row.addWidget(QLabel("模式:"))
        self.point_radio = QRadioButton("点")
        self.point_radio.setToolTip("点模式 (P)")
        self.box_radio = QRadioButton("框")
        self.box_radio.setToolTip("框模式 (B)")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.point_radio)
        self._mode_group.addButton(self.box_radio)
        self.point_radio.setChecked(True)
        row.addWidget(self.point_radio)
        row.addWidget(self.box_radio)
        row.addWidget(_vline())

        # 点正负(左右键定正负时禁用)
        self.pos_radio = QRadioButton("正点")
        self.neg_radio = QRadioButton("负点")
        self._label_group = QButtonGroup(self)
        self._label_group.addButton(self.pos_radio)
        self._label_group.addButton(self.neg_radio)
        self.pos_radio.setChecked(True)
        row.addWidget(self.pos_radio)
        row.addWidget(self.neg_radio)
        self.button_label_check = QCheckBox("左右键定正负")
        self.button_label_check.setChecked(True)
        row.addWidget(self.button_label_check)
        row.addWidget(_vline())

        self.auto_check = QCheckBox("自动预测")
        self.auto_check.setChecked(True)
        self.auto_check.setToolTip("提示变更后自动调用 predict 更新 mask")
        row.addWidget(self.auto_check)
        self.predict_btn = QPushButton("运行分割")
        self.predict_btn.setToolTip("手动触发 predict 更新 mask(自动预测关闭时使用)")
        self.predict_btn.clicked.connect(self._run_predict)
        row.addWidget(self.predict_btn)
        row.addWidget(_vline())

        self.brush_check = QCheckBox("笔刷编辑")
        self.brush_check.setToolTip("直接修改当前组 mask(仅影响导出结果)")
        row.addWidget(self.brush_check)
        row.addWidget(QLabel("半径:"))
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 200)
        self.radius_spin.setValue(20)
        row.addWidget(self.radius_spin)
        row.addWidget(_vline())

        row.addWidget(QLabel("透明度:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(128)
        self.opacity_slider.setFixedWidth(110)
        self.opacity_slider.setToolTip("mask 不透明度 (0-255)")
        row.addWidget(self.opacity_slider)
        row.addStretch()
        return row

    def _build_toolbar2(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self.undo_btn = QPushButton("撤销点")
        self.undo_btn.setToolTip("撤销当前组最后一个点 (Z)")
        self.clear_box_btn = QPushButton("清除框")
        self.clear_box_btn.setToolTip("清除当前组的框")
        self.clear_group_btn = QPushButton("清空组")
        self.clear_group_btn.setToolTip("清空当前组全部提示")
        self.del_group_btn = QPushButton("删除当前组")
        for b in (self.undo_btn, self.clear_box_btn,
                  self.clear_group_btn, self.del_group_btn):
            row.addWidget(b)
        row.addWidget(_vline())

        self.save_cur_btn = QPushButton("保存当前mask")
        self.save_cur_btn.setToolTip("保存当前组灰度 mask PNG (Ctrl+S)")
        self.save_all_btn = QPushButton("保存全部mask")
        self.save_all_btn.setToolTip("选目录, 按组保存全部 mask PNG")
        self.save_overlay_btn = QPushButton("保存叠加图")
        self.save_overlay_btn.setToolTip("原图 + 各组着色 mask 合成 PNG")
        for b in (self.save_cur_btn, self.save_all_btn, self.save_overlay_btn):
            row.addWidget(b)
        row.addWidget(_vline())

        self.export_btn = QPushButton("导出prompt")
        self.export_btn.setToolTip("本地提示导出为 prompt 文件")
        self.import_btn = QPushButton("导入prompt")
        self.import_btn.setToolTip("从 prompt 文件导入提示(需已打开图片)")
        row.addWidget(self.export_btn)
        row.addWidget(self.import_btn)
        row.addStretch()
        return row

    def _connect_signals(self) -> None:
        self.open_btn.clicked.connect(self._open_image)
        self.point_radio.toggled.connect(
            lambda on: on and self.canvas.set_mode("point"))
        self.box_radio.toggled.connect(
            lambda on: on and self.canvas.set_mode("box"))
        self.pos_radio.toggled.connect(self._on_label_radio)
        self.neg_radio.toggled.connect(self._on_label_radio)
        self.button_label_check.toggled.connect(self._on_button_label_mode)
        self.brush_check.toggled.connect(self._on_brush_toggled)
        self.radius_spin.valueChanged.connect(self._on_radius_changed)
        self.opacity_slider.valueChanged.connect(self.canvas.set_opacity)

        self.undo_btn.clicked.connect(self._undo_point)
        self.clear_box_btn.clicked.connect(self._clear_box)
        self.clear_group_btn.clicked.connect(self._clear_group)
        self.del_group_btn.clicked.connect(self._delete_current_group)
        self.save_cur_btn.clicked.connect(self._save_current_mask)
        self.save_all_btn.clicked.connect(self._save_all_masks)
        self.save_overlay_btn.clicked.connect(self._save_overlay)
        self.export_btn.clicked.connect(self._export_prompts)
        self.import_btn.clicked.connect(self._import_prompts)

        self.canvas.point_added.connect(self._on_point_added)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.mask_edited.connect(self._on_mask_edited)
        self.panel.current_changed.connect(self._on_current_changed)
        self.panel.group_removed.connect(self._on_group_removed)
        self.panel.visibility_changed.connect(
            lambda _gid, _vis: self._sync_group_display())

        # 初始交互状态同步到画布
        self.canvas.set_button_label_mode(True)
        self._on_button_label_mode(True)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("P"), self,
                  activated=lambda: self.point_radio.setChecked(True))
        QShortcut(QKeySequence("B"), self,
                  activated=lambda: self.box_radio.setChecked(True))
        QShortcut(QKeySequence("Z"), self, activated=self._undo_point)
        QShortcut(QKeySequence("Ctrl+S"), self,
                  activated=self._save_current_mask)
        for n in range(1, 10):  # 数字 1-9 切到第 n 个组
            QShortcut(QKeySequence(str(n)), self,
                      activated=lambda n=n: self._select_nth_group(n))

    # ================= 状态刷新辅助 =================
    def _group_has_prompts(self, gid: int) -> bool:
        e = self._prompts.get(gid)
        return bool(e and (e["points"] or e["box"]))

    def _has_any_prompts(self) -> bool:
        return any(e["points"] or e["box"] for e in self._prompts.values())

    def _refresh_prompts(self) -> None:
        """按本地 _prompts 重建画布提示显示(每次变更后调用)。"""
        points: List[Tuple[float, float, int, int]] = []
        boxes: List[Tuple[float, float, float, float, int]] = []
        for gid, e in self._prompts.items():
            points.extend((x, y, label, gid) for x, y, label in e["points"])
            if e["box"]:
                boxes.append((*e["box"], gid))
        self.canvas.set_prompts(points, boxes)
        self._update_status()

    def _sync_group_display(self) -> None:
        """画布组颜色/隐藏组跟随面板。"""
        self.canvas.set_group_colors(
            {gid: self.panel.color_of(gid) for gid in self.panel.groups()})
        hidden = set(self.panel.groups()) - self.panel.visible_groups()
        self.canvas.set_hidden_groups(hidden)

    def _update_buttons(self) -> None:
        has_session = self._sid is not None and not self._creating
        for b in (self.predict_btn, self.undo_btn, self.clear_box_btn,
                  self.clear_group_btn, self.del_group_btn, self.save_cur_btn,
                  self.save_all_btn, self.save_overlay_btn, self.export_btn,
                  self.import_btn):
            b.setEnabled(has_session)
        self.open_btn.setEnabled(not self._creating)

    def _update_status(self) -> None:
        sid = f"{self._sid[:8]}..." if self._sid else "无"
        pts = sum(len(e["points"]) for e in self._prompts.values())
        boxes = sum(1 for e in self._prompts.values() if e["box"])
        parts = [f"会话: {sid}", f"提示: {pts} 点 / {boxes} 框"]
        if self.brush_check.isChecked():
            parts.append("笔刷修改仅影响导出结果")
        if self._last_error:
            parts.append(f"最后错误: {self._last_error}")
        self.status_label.setText("  |  ".join(parts))

    def _show_error(self, msg: str) -> None:
        self._last_error = msg
        self._update_status()
        self._ctx.status_message.emit(f"图像页错误: {msg}")

    def _default_dir(self) -> str:
        return str(_MASKS_DIR) if _MASKS_DIR.is_dir() else str(Path.home())

    # ================= 打开图片 / 会话 =================
    def _open_image(self) -> None:
        if self._creating:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", str(Path.home()),
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp);;所有文件 (*)")
        if not path:
            return
        try:
            pil = Image.open(path)
            pil.load()
            pil = pil.convert("RGB")
        except Exception as e:
            QMessageBox.warning(self, "打开失败", f"无法读取图片:\n{e}")
            return

        if self._sid:  # 旧会话 fire-and-forget 关闭
            run_api(self._ctx.client.close_image_session, self._sid,
                    on_err=self._show_error, parent=self)
        self._reset_local_state()
        self._creating = True
        self._update_buttons()
        self._ctx.status_message.emit("正在创建图像会话...")
        run_api(self._ctx.client.create_image_session, pil,
                on_ok=lambda sid: self._on_session_created(sid, pil),
                on_err=self._on_session_err, parent=self)

    def _reset_local_state(self) -> None:
        self._sid = None
        self._pil_image = None
        self._prompts.clear()
        self._masks.clear()
        self._last_error = ""
        if self.brush_check.isChecked():
            self.brush_check.setChecked(False)  # 连带 set_edit_group(None)
        self.panel.clear()
        self.canvas.clear_masks()
        self.canvas.set_prompts([], [])
        self._sync_group_display()
        self._update_status()

    def _on_session_created(self, sid: str, pil: Image.Image) -> None:
        self._creating = False
        self._sid = sid
        self._pil_image = pil
        self.canvas.set_image(pil_to_qimage(pil))
        self._update_buttons()
        self._update_status()
        self._ctx.status_message.emit(f"图像会话已创建: {sid[:8]}...")

    def _on_session_err(self, msg: str) -> None:
        self._creating = False
        self._update_buttons()
        self._show_error(msg)
        QMessageBox.warning(self, "会话创建失败", msg)

    def shutdown(self) -> None:
        """关闭页面: 有会话则 fire-and-forget 关闭。"""
        if self._sid:
            sid, self._sid = self._sid, None
            run_api(self._ctx.client.close_image_session, sid, parent=self)

    # ================= 组管理 =================
    def _current_gid(self) -> Optional[int]:
        return self.panel.current_group()

    def _select_nth_group(self, n: int) -> None:
        groups = self.panel.groups()
        if len(groups) >= n:
            self.panel.set_current(groups[n - 1])

    def _on_current_changed(self, gid: int) -> None:
        self._sync_group_display()
        if self.brush_check.isChecked():
            # 当前组切换: 笔刷跟随(新组无 mask 则退出笔刷)
            if gid >= 0 and gid in self._masks:
                self._brush_gid = gid
                self.canvas.set_edit_group(gid, self.radius_spin.value())
            else:
                self.brush_check.setChecked(False)
        self._update_status()

    def _on_group_removed(self, gid: int) -> None:
        """面板删除组(行内 ✕ / 删除当前组按钮): 本地清理 + 通知服务端。"""
        entry = self._prompts.pop(gid, None)
        had_server_state = bool(entry and (entry["points"] or entry["box"]))
        self._masks.pop(gid, None)
        if self._brush_gid == gid:
            self.brush_check.setChecked(False)
        self.canvas.set_masks(self._masks)
        self._refresh_prompts()
        self._sync_group_display()
        if self._sid and had_server_state:
            run_api(self._ctx.client.delete_group, self._sid, gid,
                    on_ok=self._apply_server_masks,
                    on_err=self._show_error, parent=self)

    def _delete_current_group(self) -> None:
        gid = self._current_gid()
        if gid is None:
            self._ctx.status_message.emit("没有当前组可删除")
            return
        self.panel.remove_group(gid)  # 经 group_removed 走统一删除流程

    # ================= 标注交互 =================
    def _ensure_current_group(self) -> int:
        """取当前组; 无则自动新建并设为当前组。"""
        gid = self.panel.current_group()
        if gid is None:
            gid = self.panel.add_group()
            self._ctx.status_message.emit(f"已自动新建物体 {gid}")
        return gid

    def _on_point_added(self, x: float, y: float, label: int) -> None:
        if not self._sid or self._creating:
            return
        gid = self._ensure_current_group()
        entry = self._prompts.setdefault(gid, {"points": [], "box": None})
        entry["points"].append((x, y, label))
        self._refresh_prompts()
        run_api(self._ctx.client.add_point, self._sid, gid, x, y, label,
                on_ok=lambda _r: self._auto_predict(),
                on_err=lambda msg: self._on_add_point_err(gid, msg),
                parent=self)

    def _on_add_point_err(self, gid: int, msg: str) -> None:
        entry = self._prompts.get(gid)
        if entry and entry["points"]:
            entry["points"].pop()  # 服务端未记录, 回滚本地
        self._refresh_prompts()
        self._show_error(msg)

    def _on_box_drawn(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if not self._sid or self._creating:
            return
        gid = self._ensure_current_group()
        entry = self._prompts.setdefault(gid, {"points": [], "box": None})
        prev = entry["box"]
        entry["box"] = (x1, y1, x2, y2)  # 一组一个框, 新框覆盖旧框
        self._refresh_prompts()
        run_api(self._ctx.client.add_box, self._sid, gid, x1, y1, x2, y2,
                on_ok=lambda _r: self._auto_predict(),
                on_err=lambda msg: self._on_add_box_err(gid, prev, msg),
                parent=self)

    def _on_add_box_err(self, gid: int, prev, msg: str) -> None:
        entry = self._prompts.get(gid)
        if entry:
            entry["box"] = prev  # 回滚本地框
        self._refresh_prompts()
        self._show_error(msg)

    def _on_label_radio(self) -> None:
        if not self.button_label_check.isChecked():
            self.canvas.set_label(1 if self.pos_radio.isChecked() else 0)

    def _on_button_label_mode(self, on: bool) -> None:
        self.canvas.set_button_label_mode(on)
        self.pos_radio.setEnabled(not on)
        self.neg_radio.setEnabled(not on)
        if not on:
            self._on_label_radio()

    # ================= 预测 / mask =================
    def _auto_predict(self) -> None:
        """自动预测开启时跟随提示变更触发 predict。"""
        if not self.auto_check.isChecked():
            return
        self._run_predict()

    def _run_predict(self) -> None:
        """触发 predict(手动按钮或自动跟随): 有会话且有提示才发请求
        (空提示服务端会报错, 直接跳过)。"""
        if not self._sid or not self._has_any_prompts():
            return
        run_api(self._ctx.client.predict_image, self._sid,
                on_ok=self._apply_server_masks,
                on_err=self._show_error, parent=self)

    def _apply_server_masks(self, masks: Dict[int, Image.Image]) -> None:
        # 只保留本地仍有提示的组: 服务端 delete_group 可能返回已清空组的旧 mask
        self._masks = {gid: pil_to_qimage(m) for gid, m in masks.items()
                       if self._group_has_prompts(gid)}
        self.canvas.set_masks(self._masks)
        self._last_error = ""
        self._update_status()

    def _on_mask_edited(self, gid: int, qimg: QImage) -> None:
        self._masks[gid] = qimg  # 笔刷结果仅本地保存(用于导出), 不回传模型
        self._update_status()

    # ================= 撤销 / 清除 =================
    def _undo_point(self) -> None:
        if not self._sid:
            return
        gid = self._current_gid()
        if gid is None or not self._prompts.get(gid, {}).get("points"):
            self._ctx.status_message.emit("当前组没有可撤销的点")
            return
        run_api(self._ctx.client.delete_point, self._sid, gid, -1,
                on_ok=lambda _r: self._on_point_deleted(gid),
                on_err=self._show_error, parent=self)

    def _on_point_deleted(self, gid: int) -> None:
        entry = self._prompts.get(gid)
        if entry and entry["points"]:
            entry["points"].pop()
        self._refresh_prompts()
        self._auto_predict()

    def _clear_box(self) -> None:
        if not self._sid:
            return
        gid = self._current_gid()
        if gid is None or not self._prompts.get(gid, {}).get("box"):
            self._ctx.status_message.emit("当前组没有框可清除")
            return
        run_api(self._ctx.client.clear_box, self._sid, gid,
                on_ok=lambda _r: self._on_box_cleared(gid),
                on_err=self._show_error, parent=self)

    def _on_box_cleared(self, gid: int) -> None:
        entry = self._prompts.get(gid)
        if entry:
            entry["box"] = None
        self._refresh_prompts()
        self._auto_predict()

    def _clear_group(self) -> None:
        if not self._sid:
            return
        gid = self._current_gid()
        if gid is None or not self._group_has_prompts(gid):
            self._ctx.status_message.emit("当前组没有提示可清空")
            return
        run_api(self._ctx.client.clear_group, self._sid, gid,
                on_ok=lambda _r: self._on_group_cleared(gid),
                on_err=self._show_error, parent=self)

    def _on_group_cleared(self, gid: int) -> None:
        entry = self._prompts.get(gid)
        if entry:
            entry["points"] = []
            entry["box"] = None
        self._masks.pop(gid, None)  # 服务端不再返回该组 mask
        self.canvas.set_masks(self._masks)
        self._refresh_prompts()
        self._auto_predict()

    # ================= 笔刷编辑 =================
    def _on_brush_toggled(self, on: bool) -> None:
        if on:
            gid = self._current_gid()
            if gid is None:
                self._ctx.status_message.emit("请先选择要编辑的组")
                self._revert_brush_check()
                return
            if gid not in self._masks:
                self._ctx.status_message.emit("当前组暂无 mask, 请先添加提示并预测")
                self._revert_brush_check()
                return
            self._brush_gid = gid
            self.canvas.set_edit_group(gid, self.radius_spin.value())
        else:
            self._brush_gid = None
            self.canvas.set_edit_group(None)
        self._update_status()

    def _revert_brush_check(self) -> None:
        self.brush_check.blockSignals(True)
        self.brush_check.setChecked(False)
        self.brush_check.blockSignals(False)

    def _on_radius_changed(self, r: int) -> None:
        if self.brush_check.isChecked() and self._brush_gid is not None:
            self.canvas.set_edit_group(self._brush_gid, r)

    # ================= 保存 =================
    def _save_current_mask(self) -> None:
        gid = self._current_gid()
        mask = self._masks.get(gid) if gid is not None else None
        if mask is None:
            self._ctx.status_message.emit("当前组没有 mask 可保存")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存当前组 mask", str(Path(self._default_dir())
                                       / f"mask_group{gid}_{ts}.png"),
            "PNG 图片 (*.png)")
        if not path:
            return
        try:
            qimage_to_pil(mask).save(path)  # 灰度原尺寸
        except OSError as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self._ctx.status_message.emit(f"mask 已保存: {path}")

    def _save_all_masks(self) -> None:
        if not self._masks:
            self._ctx.status_message.emit("没有 mask 可保存")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "选择 mask 保存目录", self._default_dir())
        if not out_dir:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        try:
            for gid, mask in sorted(self._masks.items()):
                qimage_to_pil(mask).save(
                    str(Path(out_dir) / f"mask_group{gid}_{ts}.png"))
        except OSError as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self._ctx.status_message.emit(
            f"已保存 {len(self._masks)} 个 mask 到 {out_dir}")

    def _save_overlay(self) -> None:
        if self._pil_image is None or not self._masks:
            self._ctx.status_message.emit("没有原图或 mask, 无法保存叠加图")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存叠加图",
            str(Path(self._default_dir()) / f"overlay_{ts}.png"),
            "PNG 图片 (*.png)")
        if not path:
            return
        base = np.asarray(self._pil_image.convert("RGB"), dtype=np.float32)
        alpha = self.opacity_slider.value() / 255.0
        for gid, mask in self._masks.items():
            m = np.asarray(qimage_to_pil(mask)) > 0
            if m.shape != base.shape[:2]:
                continue  # 尺寸不符(理论不发生), 跳过
            c = self.panel.color_of(gid)
            color = np.array([c.red(), c.green(), c.blue()], dtype=np.float32)
            base[m] = base[m] * (1.0 - alpha) + color * alpha
        try:
            Image.fromarray(base.astype(np.uint8), "RGB").save(path)
        except OSError as e:
            QMessageBox.warning(self, "保存失败", str(e))
            return
        self._ctx.status_message.emit(f"叠加图已保存: {path}")

    # ================= prompt 文件 =================
    def _export_prompts(self) -> None:
        groups = []
        for gid in sorted(self._prompts):
            e = self._prompts[gid]
            if not e["points"] and not e["box"]:
                continue  # 空组不导出
            groups.append({
                "group_id": gid,
                "points": [[x, y] for x, y, _ in e["points"]],
                "labels": [label for _, _, label in e["points"]],
                "box": list(e["box"]) if e["box"] else None,
            })
        if not groups:
            self._ctx.status_message.emit("没有可导出的提示")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 prompt 文件",
            str(Path(self._default_dir()) / f"image_prompt_{ts}.json"),
            "prompt 文件 (*.json)")
        if not path:
            return
        try:
            save_image_file(path, groups)
        except OSError as e:
            QMessageBox.warning(self, "导出失败", str(e))
            return
        self._ctx.status_message.emit(f"prompt 已导出: {path}")

    def _import_prompts(self) -> None:
        if not self._sid:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 prompt 文件", self._default_dir(), "prompt 文件 (*.json)")
        if not path:
            return
        try:
            data = load_file(path)
        except ValueError as e:
            QMessageBox.warning(self, "导入失败", str(e))
            return
        if data.get("type") != "image":
            QMessageBox.warning(self, "导入失败",
                                "该文件不是图像 prompt 文件 (type != image)")
            return
        mode, ok = QInputDialog.getItem(
            self, "导入 prompt 文件",
            "合并模式:\nappend = 追加点 / 覆盖框\n"
            "replace = 整组替换\nskip = 已有组整组跳过",
            ["append", "replace", "skip"], 0, False)
        if not ok:
            return
        self._ctx.status_message.emit("正在导入 prompt 文件...")
        run_api(self._ctx.client.load_image_prompt_file, self._sid, data, mode,
                on_ok=lambda result: self._on_import_ok(result, data, mode),
                on_err=self._show_error, parent=self)

    def _on_import_ok(self, result: dict, data: dict, mode: str) -> None:
        """导入成功: 按服务端语义(append 追加/覆盖框, replace 整组替换, skip 跳过)同步本地。"""
        for g in data.get("groups", []):
            gid = int(g["group_id"])
            if mode == "skip" and self._group_has_prompts(gid):
                continue  # 服务端只认"有提示的组", 本地对齐
            entry = self._prompts.setdefault(gid, {"points": [], "box": None})
            if mode == "replace":
                entry["points"] = []
                entry["box"] = None
            for p, label in zip(g.get("points") or [], g.get("labels") or []):
                entry["points"].append((float(p[0]), float(p[1]), int(label)))
            box = g.get("box")
            if box is not None:  # append/replace: 有框则覆盖, 无框保留现状
                entry["box"] = tuple(float(v) for v in box)
            self.panel.ensure_group(gid)
        self._sync_group_display()
        self._refresh_prompts()
        skipped = result.get("groups_skipped") or []
        msg = f"prompt 导入完成(模式 {mode})"
        if skipped:
            msg += f", 跳过组 {skipped}"
        self._ctx.status_message.emit(msg)
        self._auto_predict()


def _vline() -> QFrame:
    """工具条竖直分隔线。"""
    line = QFrame()
    line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    return line
