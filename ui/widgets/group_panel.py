"""物体组管理面板: 新建/删除/重命名/可见性/当前组选择。"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from ..utils import group_color


class _GroupRow(QWidget):
    """单个物体组行: 色块 + 名称(双击可改) + 可见性勾选 + 删除按钮。"""

    clicked = Signal(int)
    remove_clicked = Signal(int)
    visibility_toggled = Signal(int, bool)
    name_edited = Signal(int, str)

    def __init__(self, gid: int, name: str, color: QColor, parent=None):
        super().__init__(parent)
        self.gid = gid
        self._name = name
        self.setObjectName("group_row")
        self.setAttribute(Qt.WA_StyledBackground, True)  # 让高亮背景生效

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)

        swatch = QLabel()
        swatch.setFixedSize(16, 16)
        swatch.setStyleSheet(f"background-color: {color.name()};"
                             "border-radius: 3px; border: 1px solid #555;")
        layout.addWidget(swatch)

        self.name_edit = QLineEdit(name)
        self.name_edit.setReadOnly(True)
        self.name_edit.setFrame(False)
        self.name_edit.installEventFilter(self)  # 单击选中行, 双击进入编辑
        self.name_edit.editingFinished.connect(self._on_editing_finished)
        layout.addWidget(self.name_edit, stretch=1)

        self.visible_check = QCheckBox()
        self.visible_check.setChecked(True)
        self.visible_check.setToolTip("显示/隐藏该组")
        self.visible_check.toggled.connect(
            lambda checked: self.visibility_toggled.emit(self.gid, checked))
        layout.addWidget(self.visible_check)

        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(22, 22)
        remove_btn.setToolTip("删除该组")
        remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self.gid))
        layout.addWidget(remove_btn)

    def set_name(self, name: str) -> None:
        self._name = name
        self.name_edit.setText(name)

    def set_highlight(self, on: bool) -> None:
        """当前行高亮(动态属性 + 重新抛光样式)。"""
        self.setProperty("current", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.gid)
        super().mousePressEvent(event)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.name_edit and self.name_edit.isReadOnly():
            if event.type() == QEvent.MouseButtonPress:
                self.clicked.emit(self.gid)  # 只读态点击视为选中行
                return True
            if event.type() == QEvent.MouseButtonDblClick:
                self.name_edit.setReadOnly(False)
                self.name_edit.selectAll()
                self.name_edit.setFocus()
                return True
        return super().eventFilter(obj, event)

    def _on_editing_finished(self) -> None:
        self.name_edit.setReadOnly(True)
        self.name_edit.deselect()
        text = self.name_edit.text().strip()
        if text and text != self._name:
            self._name = text
            self.name_edited.emit(self.gid, text)
        else:
            self.name_edit.setText(self._name)  # 空名/未改: 还原


class GroupPanel(QWidget):
    """物体组管理面板(新建/删除/重命名/可见性/当前组)。

    无当前组时 current_changed 发 -1(Signal(int) 无法携带 None)。
    """

    current_changed = Signal(int)
    group_removed = Signal(int)
    visibility_changed = Signal(int, bool)
    renamed = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups: Dict[int, dict] = {}  # gid -> {"name", "color", "visible"}
        self._rows: Dict[int, _GroupRow] = {}
        self._current: Optional[int] = None
        self._next_id = 0  # 单调递增计数器, 避免删除后撞 id
        self._setup_ui()

    # ---------------- 组管理 ----------------
    def add_group(self, name: Optional[str] = None) -> int:
        """新建组(自动分配 id=现有最大+1, 自动配色, 默认名 "物体 N"), 返回 gid。"""
        gid = max(self._next_id, max(self._groups.keys(), default=-1) + 1)
        self._next_id = gid + 1
        self._create(gid, name or f"物体 {gid}")
        self.set_current(gid)
        return gid

    def ensure_group(self, gid: int) -> int:
        """导入用: 不存在则以指定 id 创建, 并推进内部计数器避免后续撞 id。"""
        gid = int(gid)
        if gid not in self._groups:
            self._create(gid, f"物体 {gid}")
        self._next_id = max(self._next_id, gid + 1)
        return gid

    def remove_group(self, gid: int) -> None:
        """删除组并发 group_removed; 当前组被删时自动切到剩余第一组。"""
        if gid not in self._groups:
            return
        del self._groups[gid]
        row = self._rows.pop(gid)
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self.group_removed.emit(gid)
        if self._current == gid:
            remaining = sorted(self._groups.keys())
            if remaining:
                self.set_current(remaining[0])
            else:
                self._current = None
                self.current_changed.emit(-1)

    def clear(self) -> None:
        """清空全部组(每组都发 group_removed)。"""
        for gid in list(self._groups.keys()):
            self.remove_group(gid)

    # ---------------- 查询 ----------------
    def groups(self) -> List[int]:
        return sorted(self._groups.keys())

    def current_group(self) -> Optional[int]:
        return self._current

    def color_of(self, gid: int) -> QColor:
        info = self._groups.get(gid)
        return info["color"] if info else group_color(gid)

    def is_visible(self, gid: int) -> bool:
        return self._groups[gid]["visible"]

    def visible_groups(self) -> Set[int]:
        return {gid for gid, info in self._groups.items() if info["visible"]}

    def name_of(self, gid: int) -> str:
        return self._groups[gid]["name"]

    # ---------------- 修改 ----------------
    def set_current(self, gid: int) -> None:
        """设置当前组(发 current_changed); 传 None 清除当前组(发 -1)。"""
        if gid is None:
            if self._current is not None:
                self._current = None
                self._update_highlight()
                self.current_changed.emit(-1)
            return
        if gid not in self._groups or gid == self._current:
            return
        self._current = gid
        self._update_highlight()
        self.current_changed.emit(gid)

    def set_name(self, gid: int, name: str) -> None:
        """重命名(名称实际变化时发 renamed)。"""
        info = self._groups.get(gid)
        name = (name or "").strip()
        if info is None or not name or info["name"] == name:
            return
        info["name"] = name
        self._rows[gid].set_name(name)
        self.renamed.emit(gid, name)

    # ---------------- 内部 ----------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.add_btn = QPushButton("＋ 新建物体")
        self.add_btn.clicked.connect(lambda: self.add_group())
        layout.addWidget(self.add_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        self._rows_layout = QVBoxLayout(container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, stretch=1)

        self.setStyleSheet("""
            #group_row { border-radius: 4px; }
            #group_row[current="true"] { background-color: #3a3d41; }
            QLineEdit { background: transparent; color: #e0e0e0; }
            QCheckBox { color: #e0e0e0; }
        """)

    def _create(self, gid: int, name: str) -> None:
        self._groups[gid] = {"name": name,
                             "color": group_color(gid), "visible": True}
        row = _GroupRow(gid, name, self._groups[gid]["color"])
        row.clicked.connect(self.set_current)
        row.remove_clicked.connect(self.remove_group)
        row.visibility_toggled.connect(self._on_visibility)
        row.name_edited.connect(self._on_name_edited)
        self._rows[gid] = row
        # 按 gid 排序插入(stretch 恒在最后)
        pos = sorted(self._groups.keys()).index(gid)
        self._rows_layout.insertWidget(pos, row)
        row.show()

    def _update_highlight(self) -> None:
        for gid, row in self._rows.items():
            row.set_highlight(gid == self._current)

    def _on_visibility(self, gid: int, checked: bool) -> None:
        if gid in self._groups:
            self._groups[gid]["visible"] = checked
            self.visibility_changed.emit(gid, checked)

    def _on_name_edited(self, gid: int, text: str) -> None:
        self.set_name(gid, text)
