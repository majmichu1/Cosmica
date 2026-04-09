"""Batch Processing — apply processing pipelines to multiple images.

Supports defining ordered processing steps and applying them to a batch
of input files with parallel execution support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cosmica.core.image_io import ImageData, load_image, save_fits

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class PipelineStep:
    """A single processing step in a batch pipeline."""

    tool_name: str  # registered tool name (e.g., "auto_stretch", "denoise")
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class Pipeline:
    """An ordered list of processing steps."""

    name: str = "Untitled Pipeline"
    steps: list[PipelineStep] = field(default_factory=list)

    def add_step(self, tool_name: str, params: dict[str, Any] | None = None) -> PipelineStep:
        step = PipelineStep(tool_name=tool_name, params=params or {})
        self.steps.append(step)
        return step

    def remove_step(self, index: int):
        if 0 <= index < len(self.steps):
            self.steps.pop(index)

    def move_step(self, from_index: int, to_index: int):
        if 0 <= from_index < len(self.steps) and 0 <= to_index < len(self.steps):
            step = self.steps.pop(from_index)
            self.steps.insert(to_index, step)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "steps": [
                {"tool_name": s.tool_name, "params": s.params, "enabled": s.enabled}
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Pipeline:
        pipeline = cls(name=d.get("name", "Untitled"))
        for step_d in d.get("steps", []):
            step = PipelineStep(
                tool_name=step_d["tool_name"],
                params=step_d.get("params", {}),
                enabled=step_d.get("enabled", True),
            )
            pipeline.steps.append(step)
        return pipeline


# Tool registry: maps tool names to processing functions
_TOOL_REGISTRY: dict[str, Callable] = {}


def register_tool(name: str, func: Callable):
    """Register a processing function for use in batch pipelines."""
    _TOOL_REGISTRY[name] = func


def get_registered_tools() -> dict[str, Callable]:
    """Get all registered tools."""
    return dict(_TOOL_REGISTRY)


def _register_default_tools():
    """Register all built-in processing tools."""
    from cosmica.core.stretch import auto_stretch, generalized_hyperbolic_stretch
    from cosmica.core.background import extract_background
    from cosmica.core.cosmetic import cosmetic_correction
    from cosmica.core.banding import banding_reduction
    from cosmica.core.histogram_transform import histogram_transform
    from cosmica.core.color_tools import scnr, color_adjust
    from cosmica.core.denoise import denoise
    from cosmica.core.local_contrast import local_contrast_enhance

    register_tool("auto_stretch", lambda data, **kw: auto_stretch(data))
    register_tool("ghs", lambda data, **kw: generalized_hyperbolic_stretch(data))
    register_tool("background_extraction", lambda data, **kw: extract_background(data)[0])
    register_tool("cosmetic_correction", lambda data, **kw: cosmetic_correction(data).data)
    register_tool("banding_reduction", banding_reduction)
    register_tool("histogram_transform", histogram_transform)
    register_tool("scnr", scnr)
    register_tool("color_adjust", color_adjust)
    register_tool("denoise", denoise)
    register_tool("local_contrast", local_contrast_enhance)


def apply_pipeline_to_image(
    data: np.ndarray,
    pipeline: Pipeline,
) -> np.ndarray:
    """Apply a pipeline to a single image.

    Parameters
    ----------
    data : ndarray
        Image data, float32 in [0, 1].
    pipeline : Pipeline
        Processing pipeline to apply.

    Returns
    -------
    ndarray
        Processed image.
    """
    if not _TOOL_REGISTRY:
        _register_default_tools()

    result = data.copy()
    for step in pipeline.steps:
        if not step.enabled:
            continue
        func = _TOOL_REGISTRY.get(step.tool_name)
        if func is None:
            log.warning("Unknown tool: %s, skipping", step.tool_name)
            continue
        log.info("Applying: %s", step.tool_name)
        result = func(result, **step.params)

    return result


@dataclass
class BatchResult:
    """Result of batch processing."""

    n_processed: int
    n_failed: int
    output_paths: list[Path]
    errors: list[str]


def batch_process(
    input_paths: list[Path],
    pipeline: Pipeline,
    output_dir: Path,
    output_format: str = "fits",
    progress: ProgressCallback | None = None,
) -> BatchResult:
    """Apply a pipeline to multiple input files.

    Parameters
    ----------
    input_paths : list[Path]
        Input image file paths.
    pipeline : Pipeline
        Processing pipeline to apply.
    output_dir : Path
        Directory to save processed images.
    output_format : str
        Output format ("fits" or "xisf").
    progress : callable, optional
        Progress callback.

    Returns
    -------
    BatchResult
        Processing results.
    """
    if progress is None:
        progress = _noop_progress

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    errors: list[str] = []
    n_processed = 0

    for i, path in enumerate(input_paths):
        frac = i / max(len(input_paths), 1)
        progress(frac, f"Processing {path.name} ({i + 1}/{len(input_paths)})...")

        try:
            img = load_image(str(path))
            processed = apply_pipeline_to_image(img.data, pipeline)

            out_name = f"{path.stem}_processed.{output_format}"
            out_path = output_dir / out_name
            save_fits(ImageData(data=processed, header=img.header), out_path)

            output_paths.append(out_path)
            n_processed += 1
        except Exception as e:
            log.error("Error processing %s: %s", path.name, e)
            errors.append(f"{path.name}: {e}")

    progress(1.0, f"Batch complete: {n_processed}/{len(input_paths)} processed")

    return BatchResult(
        n_processed=n_processed,
        n_failed=len(errors),
        output_paths=output_paths,
        errors=errors,
    )
