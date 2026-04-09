"""Preset system — save/load tool parameter presets as JSON.

Supports all 34 *Params dataclasses across cosmica/core/.
Handles Enums, numpy arrays, nested dataclasses, and lists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Preset directories: system-bundled + user-created
_SYSTEM_PRESET_DIR = Path(__file__).resolve().parent.parent.parent / "presets"
_USER_PRESET_DIR = (
    Path(__import__("platformdirs").user_data_dir("Cosmica", "Cosmica")) / "presets"
    if __import__("platformdirs").user_data_dir("Cosmica", "Cosmica")
    else Path.home() / ".local" / "share" / "Cosmica" / "presets"
)


def get_user_preset_dir() -> Path:
    """Return the user preset directory, creating it if needed."""
    try:
        from platformdirs import user_data_dir
        directory = Path(user_data_dir("Cosmica", "Cosmica")) / "presets"
    except ImportError:
        directory = Path.home() / ".local" / "share" / "Cosmica" / "presets"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_system_preset_dir() -> Path:
    """Return the system-bundled preset directory."""
    return _SYSTEM_PRESET_DIR


def _serialize_value(value: Any) -> Any:
    """Serialize a single value to JSON-safe format."""
    if isinstance(value, Enum):
        return {"__enum__": value.__class__.__name__, "name": value.name}
    elif isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, tuple):
        return {"__tuple__": list(value)}
    elif isinstance(value, list):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    return value


def _deserialize_value(value: Any, field_type: type | None = None) -> Any:
    """Deserialize a JSON-safe value back to its original type."""
    if isinstance(value, dict):
        if "__enum__" in value:
            enum_name = value["__enum__"]
            # Import the enum class from the appropriate module
            enum_class = _get_enum_class(enum_name)
            if enum_class:
                return enum_class[value["name"]]
            return value["name"]  # fallback
        if "__ndarray__" in value:
            dtype = value.get("dtype", "float32")
            return np.array(value["__ndarray__"], dtype=np.dtype(dtype))
        if "__tuple__" in value:
            return tuple(value["__tuple__"])
        return {k: _deserialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_deserialize_value(v) for v in value]
    return value


def _get_enum_class(enum_name: str) -> type[Enum] | None:
    """Look up an Enum class by name from cosmica/core modules."""
    enum_map = {
        "RejectionMethod": "cosmica.core.stacking",
        "IntegrationMethod": "cosmica.core.stacking",
        "DenoiseMethod": "cosmica.core.denoise",
        "SCNRMethod": "cosmica.core.color_tools",
        "HDRMethod": "cosmica.core.hdr",
        "MorphOp": "cosmica.core.morphology",
        "StructuringElement": "cosmica.core.morphology",
        "NarrowbandPalette": "cosmica.core.narrowband",
        "BlendMethod": "cosmica.core.mosaic",
        "RotateAngle": "cosmica.core.transforms",
        "FlipAxis": "cosmica.core.transforms",
        "InterpolationMethod": "cosmica.core.transforms",
        "BinMode": "cosmica.core.transforms",
    }
    module_path = enum_map.get(enum_name)
    if module_path:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, enum_name, None)
    return None


def serialize_params(params: Any) -> dict:
    """Serialize a Params dataclass to a JSON-safe dict."""
    if not is_dataclass(params):
        raise TypeError(f"Expected a dataclass, got {type(params)}")

    data = asdict(params)
    return {k: _serialize_value(v) for k, v in data.items()}


# Well-known nested dataclass types used inside Params classes.
# Maps type name → (module_path, class_name) for runtime resolution.
NESTED_DATACLASS_TYPES = {
    "CurvePoints": ("cosmica.core.curves", "CurvePoints"),
}


def _resolve_type(type_hint: str | type) -> type | None:
    """Resolve a type hint (possibly a string annotation) to an actual type."""
    if isinstance(type_hint, type):
        return type_hint
    if isinstance(type_hint, str):
        # Check nested dataclass types first
        if type_hint in NESTED_DATACLASS_TYPES:
            mod_path, cls_name = NESTED_DATACLASS_TYPES[type_hint]
            import importlib
            mod = importlib.import_module(mod_path)
            return getattr(mod, cls_name, None)
    return None


def deserialize_params(cls: type, data: dict) -> Any:
    """Deserialize a dict back into a Params dataclass instance."""
    if not is_dataclass(cls):
        raise TypeError(f"Expected a dataclass class, got {type(cls)}")

    field_infos = {f.name: f for f in fields(cls)}
    deserialized = {}
    for key, value in data.items():
        raw_type = field_infos[key].type if key in field_infos else None
        actual_type = _resolve_type(raw_type)
        deserialized[key] = _deserialize_value(value, actual_type)

    # Handle nested dataclass fields that were stored as dicts
    for f in fields(cls):
        if f.name not in deserialized:
            continue
        val = deserialized[f.name]
        if isinstance(val, dict):
            actual_type = _resolve_type(f.type)
            if actual_type and is_dataclass(actual_type):
                deserialized[f.name] = _deserialize_nested_dataclass(actual_type, val)

    return cls(**deserialized)


def _deserialize_nested_dataclass(cls: type, data: dict) -> Any:
    """Deserialize a nested dataclass from a dict."""
    field_infos = {f.name: f for f in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        raw_type = field_infos[key].type if key in field_infos else None
        actual_type = _resolve_type(raw_type)
        kwargs[key] = _deserialize_value(value, actual_type)
    return cls(**kwargs)


# ── Preset registry ───────────────────────────────────────────────

# Mapping of tool name → Params class
TOOL_PARAM_CLASSES: dict[str, type] = {}


def register_tool(tool_name: str, params_class: type) -> None:
    """Register a tool's Params class for preset serialization."""
    TOOL_PARAM_CLASSES[tool_name] = params_class
    log.debug("Registered tool: %s -> %s", tool_name, params_class.__name__)


def save_preset(
    tool_name: str,
    preset_name: str,
    params: Any,
    description: str = "",
) -> Path:
    """Save a preset for a specific tool.

    Args:
        tool_name: Registered tool name (e.g. "stretch", "denoise").
        preset_name: Human-readable preset name (e.g. "Nebula Default").
        params: Params dataclass instance to save.
        description: Optional description of the preset.

    Returns:
        Path to the saved preset file.
    """
    if tool_name not in TOOL_PARAM_CLASSES:
        raise ValueError(f"Unknown tool: {tool_name}. Register it first.")

    preset_dir = get_user_preset_dir() / tool_name
    preset_dir.mkdir(parents=True, exist_ok=True)

    preset_data = {
        "tool": tool_name,
        "name": preset_name,
        "description": description,
        "version": "1.0",
        "params": serialize_params(params),
    }

    # Sanitize filename
    safe_name = preset_name.replace(" ", "_").replace("/", "_").lower()
    path = preset_dir / f"{safe_name}.json"
    with open(path, "w") as f:
        json.dump(preset_data, f, indent=2)

    log.info("Saved preset: %s for tool %s at %s", preset_name, tool_name, path)
    return path


def load_preset(tool_name: str, preset_name: str) -> Any | None:
    """Load a preset for a specific tool.

    Returns:
        Params dataclass instance, or None if not found.
    """
    cls = TOOL_PARAM_CLASSES.get(tool_name)
    if cls is None:
        raise ValueError(f"Unknown tool: {tool_name}")

    safe_name = preset_name.replace(" ", "_").replace("/", "_").lower()

    # Search user presets first, then system presets
    search_dirs = [
        get_user_preset_dir() / tool_name,
        get_system_preset_dir() / tool_name,
    ]

    for preset_dir in search_dirs:
        path = preset_dir / f"{safe_name}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return deserialize_params(cls, data["params"])

    return None


def list_presets(tool_name: str | None = None) -> list[dict]:
    """List available presets.

    Args:
        tool_name: If given, only list presets for this tool.
                   If None, list all presets.

    Returns:
        List of dicts with keys: tool, name, description, source (user/system).
    """
    results = []

    if tool_name:
        tools = [tool_name]
    else:
        # Discover all tool directories
        tools = set()
        for base in [get_user_preset_dir(), get_system_preset_dir()]:
            if base.exists():
                for child in base.iterdir():
                    if child.is_dir():
                        tools.add(child.name)

    for tool in sorted(tools):
        for preset_dir, source in [
            (get_user_preset_dir() / tool, "user"),
            (get_system_preset_dir() / tool, "system"),
        ]:
            if not preset_dir.exists():
                continue
            for path in sorted(preset_dir.glob("*.json")):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    results.append({
                        "tool": data.get("tool", tool),
                        "name": data.get("name", path.stem),
                        "description": data.get("description", ""),
                        "source": source,
                        "path": str(path),
                    })
                except (json.JSONDecodeError, KeyError):
                    log.warning("Invalid preset file: %s", path)

    return results


def delete_preset(tool_name: str, preset_name: str) -> bool:
    """Delete a user preset. System presets are read-only.

    Returns:
        True if deleted, False if not found or system preset.
    """
    safe_name = preset_name.replace(" ", "_").replace("/", "_").lower()
    path = get_user_preset_dir() / tool_name / f"{safe_name}.json"
    if path.exists():
        path.unlink()
        log.info("Deleted preset: %s for tool %s", preset_name, tool_name)
        return True
    return False


def load_default_presets() -> None:
    """Register all built-in tools with their Params classes."""
    from cosmica.core.abe import ABEParams
    from cosmica.core.background import BackgroundParams
    from cosmica.core.banding import BandingParams
    from cosmica.core.chromatic_aberration import CAParams
    from cosmica.core.color_calibration import ColorCalibrationParams
    from cosmica.core.color_tools import ColorAdjustParams, SCNRParams
    from cosmica.core.cosmetic import CosmeticParams
    from cosmica.core.curves import CurvesParams
    from cosmica.core.deconvolution import DeconvolutionParams, SpatialDeconvParams
    from cosmica.core.denoise import DenoiseParams
    from cosmica.core.drizzle import DrizzleParams
    from cosmica.core.filters import MedianFilterParams, UnsharpMaskParams
    from cosmica.core.hdr import HDRParams
    from cosmica.core.histogram_transform import HistogramTransformParams
    from cosmica.core.local_contrast import LocalContrastParams
    from cosmica.core.morphology import MorphologyParams
    from cosmica.core.mosaic import MosaicParams
    from cosmica.core.narrowband import NarrowbandParams
    from cosmica.core.stacking import StackingParams
    from cosmica.core.star_reduction import StarReductionParams
    from cosmica.core.stretch import GHSParams, StretchParams
    from cosmica.core.subframe_selector import SubframeSelectorParams
    from cosmica.core.transforms import (
        BinParams,
        CropParams,
        FlipParams,
        ResizeParams,
        RotateParams,
    )
    from cosmica.core.vignette import VignetteParams
    from cosmica.core.wavelets import WaveletParams

    registrations = {
        "abe": ABEParams,
        "banding": BandingParams,
        "background": BackgroundParams,
        "chromatic_aberration": CAParams,
        "color_calibration": ColorCalibrationParams,
        "color_adjust": ColorAdjustParams,
        "scnr": SCNRParams,
        "cosmetic": CosmeticParams,
        "curves": CurvesParams,
        "deconvolution": DeconvolutionParams,
        "spatial_deconvolution": SpatialDeconvParams,
        "denoise": DenoiseParams,
        "drizzle": DrizzleParams,
        "unsharp_mask": UnsharpMaskParams,
        "median_filter": MedianFilterParams,
        "hdr": HDRParams,
        "histogram_transform": HistogramTransformParams,
        "local_contrast": LocalContrastParams,
        "morphology": MorphologyParams,
        "mosaic": MosaicParams,
        "narrowband": NarrowbandParams,
        "stacking": StackingParams,
        "star_reduction": StarReductionParams,
        "stretch": StretchParams,
        "ghs": GHSParams,
        "subframe_selector": SubframeSelectorParams,
        "crop": CropParams,
        "rotate": RotateParams,
        "flip": FlipParams,
        "resize": ResizeParams,
        "bin": BinParams,
        "vignette": VignetteParams,
        "wavelets": WaveletParams,
    }

    for name, cls in registrations.items():
        register_tool(name, cls)
