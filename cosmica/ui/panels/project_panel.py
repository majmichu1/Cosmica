"""Project Panel — left sidebar showing project files and history."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cosmica.core.image_io import FrameType
from cosmica.core.project import Project


FRAME_TYPE_LABELS = {
    FrameType.LIGHT: "Light Frames",
    FrameType.DARK: "Dark Frames",
    FrameType.FLAT: "Flat Frames",
    FrameType.BIAS: "Bias / Offset",
    FrameType.MASTER_DARK: "Master Dark",
    FrameType.MASTER_FLAT: "Master Flat",
    FrameType.MASTER_BIAS: "Master Bias",
    FrameType.ALIGNED: "Aligned Frames",
    FrameType.RESULT: "Results",
}

# Groups shown as top-level sections in the file tree
_PANEL_SECTIONS = [
    ("RAW", [FrameType.LIGHT, FrameType.DARK, FrameType.FLAT, FrameType.BIAS]),
    ("CALIBRATION MASTERS", [
        FrameType.MASTER_DARK, FrameType.MASTER_FLAT, FrameType.MASTER_BIAS,
    ]),
    ("REGISTERED", [FrameType.ALIGNED]),
    ("PROCESSED", [FrameType.RESULT]),
]

FILE_FILTERS = "FITS Files (*.fit *.fits *.fts);;XISF Files (*.xisf);;TIFF Files (*.tif *.tiff);;All Files (*)"


class ProjectPanel(QWidget):
    """Left panel: file tree, import controls, processing history."""

    frame_selected = pyqtSignal(str)  # file path
    frames_imported = pyqtSignal(list, object)  # paths, FrameType

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(400)

        self._project: Project | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Project name
        self._title = QLabel("No Project")
        self._title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")
        layout.addWidget(self._title)

        # Tab widget: Files / History
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # --- Files tab ---
        files_widget = QWidget()
        files_layout = QVBoxLayout(files_widget)
        files_layout.setContentsMargins(0, 4, 0, 0)

        # Smart auto-import button
        btn_auto = QPushButton("Auto-Import Folder…")
        btn_auto.setToolTip(
            "Scan a folder for FITS/XISF files and automatically assign frame types\n"
            "from IMAGETYP/FRAME headers, filename patterns, and exposure time."
        )
        btn_auto.clicked.connect(self._import_folder_auto)
        btn_auto.setStyleSheet("padding: 4px 8px; font-size: 11px; font-weight: bold;")
        files_layout.addWidget(btn_auto)

        # Manual import buttons
        btn_row = QHBoxLayout()
        for label, frame_type in [
            ("Lights", FrameType.LIGHT),
            ("Darks", FrameType.DARK),
            ("Flats", FrameType.FLAT),
            ("Bias", FrameType.BIAS),
        ]:
            btn = QPushButton(f"+ {label}")
            btn.setToolTip(f"Import {label.lower()} frames into the project")
            btn.clicked.connect(lambda checked, ft=frame_type: self._import_frames(ft))
            btn.setProperty("flat", True)
            btn.setObjectName("flatButton")
            btn.setStyleSheet("padding: 4px 8px; font-size: 11px;")
            btn_row.addWidget(btn)
        files_layout.addLayout(btn_row)

        # File tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Name", "Info"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.setAlternatingRowColors(True)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._context_menu)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        files_layout.addWidget(self._tree)

        tabs.addTab(files_widget, "Files")

        # --- History tab ---
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(0, 4, 0, 0)

        self._history_tree = QTreeWidget()
        self._history_tree.setHeaderLabels(["Step", "Time"])
        self._history_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        history_layout.addWidget(self._history_tree)

        tabs.addTab(history_widget, "History")

    def set_project(self, project: Project | None):
        self._project = project
        if project:
            self._title.setText(project.name)
        else:
            self._title.setText("No Project")
        self.refresh()

    def refresh(self):
        self._tree.clear()
        self._history_tree.clear()

        if self._project is None:
            return

        # Build file tree grouped into pipeline sections
        for section_label, frame_types in _PANEL_SECTIONS:
            all_frames = []
            for ft in frame_types:
                all_frames.extend(self._project.frames_by_type(ft))
            if not all_frames:
                continue

            section_item = QTreeWidgetItem(self._tree, [f"── {section_label} ──", ""])
            section_item.setExpanded(True)
            section_item.setFlags(section_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            font = section_item.font(0)
            font.setBold(True)
            section_item.setFont(0, font)

            for ft in frame_types:
                frames = self._project.frames_by_type(ft)
                if not frames:
                    continue
                type_label = FRAME_TYPE_LABELS.get(ft, ft.name)
                group = QTreeWidgetItem(section_item, [f"{type_label} ({len(frames)})", ""])
                group.setExpanded(True)
                for entry in frames:
                    name = entry.path.name
                    info = ""
                    if entry.metadata.get("exposure"):
                        info = f"{entry.metadata['exposure']}s"
                    item = QTreeWidgetItem(group, [name, info])
                    item.setData(0, Qt.ItemDataRole.UserRole, str(entry.path))
                    item.setToolTip(0, str(entry.path))

        # Build history
        for step in self._project.history:
            ts = step.timestamp.split("T")[-1][:8] if "T" in step.timestamp else step.timestamp
            QTreeWidgetItem(self._history_tree, [step.name, ts])

    def _import_frames(self, frame_type: FrameType):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Import {frame_type.name.title()} Frames",
            "",
            FILE_FILTERS,
        )
        if paths:
            self.frames_imported.emit([Path(p) for p in paths], frame_type)

    def _import_folder_auto(self):
        """Scan a folder, auto-detect frame types from headers, import all."""
        from PyQt6.QtWidgets import QMessageBox
        from pathlib import Path
        import glob as _glob
        from cosmica.core.image_io import _guess_frame_type

        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Auto-Import", ""
        )
        if not folder:
            return

        folder = Path(folder)
        extensions = ("*.fits", "*.fit", "*.fts", "*.xisf",
                       "*.FITS", "*.FIT", "*.FTS", "*.XISF")
        files = []
        for ext in extensions:
            files.extend(folder.glob(ext))
            files.extend(folder.rglob(ext))
        files = sorted(set(files))

        if not files:
            QMessageBox.information(self, "Auto-Import",
                                    f"No FITS/XISF files found in:\n{folder}")
            return

        # Group by detected type
        groups: dict[FrameType, list[Path]] = {}
        unknown: list[Path] = []

        for path in files:
            try:
                # Read just the header (fast — no pixel data)
                header = {}
                if path.suffix.lower() in (".fits", ".fit", ".fts"):
                    try:
                        from astropy.io import fits as _fits
                        with _fits.open(str(path), memmap=True) as hdul:
                            for hdu in hdul:
                                if hdu.data is not None:
                                    header = dict(hdu.header)
                                    break
                    except Exception:
                        pass
                ft = _guess_frame_type(header, path)
                if ft == FrameType.UNKNOWN:
                    unknown.append(path)
                else:
                    groups.setdefault(ft, []).append(path)
            except Exception:
                unknown.append(path)

        # Emit per type group
        total = 0
        for ft, paths in groups.items():
            self.frames_imported.emit(paths, ft)
            total += len(paths)

        # Unknown frames — ask user
        if unknown:
            reply = QMessageBox.question(
                self,
                "Auto-Import — Unknown Frames",
                f"{len(unknown)} frame(s) could not be auto-detected.\n"
                f"Import them as Light frames?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.frames_imported.emit(unknown, FrameType.LIGHT)
                total += len(unknown)

        summary = ", ".join(
            f"{len(v)} {FRAME_TYPE_LABELS.get(k, k.name)}"
            for k, v in groups.items()
        )
        QMessageBox.information(
            self, "Auto-Import Complete",
            f"Imported {total} frames:\n{summary or 'none detected'}"
            + (f"\n{len(unknown)} unknown" if unknown else "")
        )

    def _context_menu(self, pos):
        item = self._tree.itemAt(pos)
        if item is None:
            return
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path is None:
            return

        menu = QMenu(self)
        open_action = menu.addAction("Open in Canvas")
        remove_action = menu.addAction("Remove from Project")

        action = menu.exec(self._tree.mapToGlobal(pos))
        if action == open_action:
            self.frame_selected.emit(path)
        elif action == remove_action:
            if self._project:
                self._project.remove_frame(Path(path))
                self.refresh()

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            self.frame_selected.emit(path)
