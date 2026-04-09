"""Image Statistics Dialog — display comprehensive per-channel statistics."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from cosmica.core.statistics import ImageStatistics


class StatisticsDialog(QDialog):
    """Dialog showing per-channel image statistics in a table."""

    def __init__(self, stats: ImageStatistics, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Statistics")
        self.setMinimumSize(620, 400)

        layout = QVBoxLayout(self)

        # Summary label
        linear_str = "Linear (unstretched)" if stats.is_linear else "Non-linear (stretched)"
        summary = QLabel(
            f"{stats.width} x {stats.height} | {stats.n_channels} channel(s) | "
            f"{stats.total_pixels:,} pixels | {linear_str}"
        )
        summary.setStyleSheet("font-size: 12px; color: #80c0ff; margin-bottom: 8px;")
        layout.addWidget(summary)

        # Table
        columns = [
            "Channel", "Mean", "Median", "Std Dev", "Min", "Max",
            "MAD", "SNR est.", "P01", "P99", "Clip Lo%", "Clip Hi%",
        ]
        n_rows = len(stats.channels)
        table = QTableWidget(n_rows, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row, ch in enumerate(stats.channels):
            values = [
                ch.name,
                f"{ch.mean:.6f}",
                f"{ch.median:.6f}",
                f"{ch.std:.6f}",
                f"{ch.min_val:.6f}",
                f"{ch.max_val:.6f}",
                f"{ch.mad:.6f}",
                f"{ch.snr_estimate:.1f}",
                f"{ch.percentile_01:.6f}",
                f"{ch.percentile_99:.6f}",
                f"{ch.clipped_low_pct:.2f}",
                f"{ch.clipped_high_pct:.2f}",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft if col == 0
                    else Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                table.setItem(row, col, item)

        layout.addWidget(table)
