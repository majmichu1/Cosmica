"""Undo/Redo system — QUndoStack-based command history for image operations."""

from __future__ import annotations

import logging

from PyQt6.QtGui import QUndoCommand, QUndoStack

from cosmica.core.image_io import ImageData

log = logging.getLogger(__name__)

# Maximum number of undo steps to keep in memory
MAX_UNDO_DEPTH = 50


class ImageEditCommand(QUndoCommand):
    """Generic undo/redo command for image transformations.

    Stores a copy of the image before and after the operation.
    For large images, this uses significant memory — the stack
    depth is limited by MAX_UNDO_DEPTH.
    """

    def __init__(
        self,
        before: ImageData,
        after: ImageData,
        description: str,
        parent: QUndoCommand | None = None,
    ):
        super().__init__(description, parent)
        self._before = _safe_copy(before)
        self._after = _safe_copy(after)
        self._target_ref: list[ImageData] | None = None  # set via set_target()

    def set_target(self, target_ref: list) -> None:
        """Set the mutable reference where undo/redo will swap the image.

        Args:
            target_ref: A single-element list [current_image] that acts as
                        a mutable reference to the main window's image state.
        """
        self._target_ref = target_ref

    def undo(self) -> None:
        if self._target_ref is not None:
            self._target_ref[0] = self._before
            log.debug("Undo: %s", self.text())

    def redo(self) -> None:
        if self._target_ref is not None:
            self._target_ref[0] = self._after
            log.debug("Redo: %s", self.text())


def _safe_copy(image: ImageData) -> ImageData:
    """Create a deep copy of ImageData for undo storage."""
    return ImageData(
        data=image.data.copy(),
        header=dict(image.header),
        file_path=image.file_path,
        frame_type=image.frame_type,
    )


class CosmicaUndoStack:
    """High-level undo/redo manager wrapping QUndoStack.

    Provides a simple API:
    - push(before, after, description) — record an operation
    - undo() / redo() — navigate history
    - can_undo() / can_redo() — query state
    - clear() — reset history
    """

    def __init__(self, max_depth: int = MAX_UNDO_DEPTH):
        self._stack = QUndoStack()
        self._max_depth = max_depth
        self._target_ref: list[ImageData | None] = [None]

    def set_target(self, target_ref: list) -> None:
        """Set the mutable reference to the current image.

        Args:
            target_ref: A single-element list [current_image].
        """
        self._target_ref = target_ref

    def push(
        self,
        before: ImageData,
        after: ImageData,
        description: str,
    ) -> None:
        """Push an operation onto the undo stack.

        Args:
            before: Image state before the operation.
            after: Image state after the operation.
            description: Human-readable description (e.g. "Stretch").
        """
        # Limit stack depth
        if self._stack.count() >= self._max_depth:
            # Force clean to prevent memory blow-up
            self._stack.setUndoLimit(self._max_depth)

        cmd = ImageEditCommand(before, after, description)
        cmd.set_target(self._target_ref)
        self._stack.push(cmd)
        log.debug("Pushed undo command: %s (stack size: %d)", description, self._stack.count())

    def undo(self) -> bool:
        """Undo the last operation. Returns True if successful."""
        if self.can_undo():
            self._stack.undo()
            return True
        return False

    def redo(self) -> bool:
        """Redo the last undone operation. Returns True if successful."""
        if self.can_redo():
            self._stack.redo()
            return True
        return False

    def can_undo(self) -> bool:
        return self._stack.canUndo()

    def can_redo(self) -> bool:
        return self._stack.canRedo()

    def clear(self) -> None:
        self._stack.clear()
        log.debug("Undo stack cleared")

    @property
    def count(self) -> int:
        return self._stack.count()

    def undo_text(self) -> str:
        """Return description of the next undo action."""
        return self._stack.undoText()

    def redo_text(self) -> str:
        """Return description of the next redo action."""
        return self._stack.redoText()
