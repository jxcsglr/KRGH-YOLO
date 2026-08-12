"""Metric and run-summary helpers."""

import json
import platform
from pathlib import Path


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_f1(precision, recall):
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return 2 * precision * recall / (precision + recall)


def metric_summary(metrics):
    precision = to_float(metrics.box.mp)
    recall = to_float(metrics.box.mr)
    speed = {
        key: to_float(value) for key, value in getattr(metrics, "speed", {}).items()
    }
    return {
        "precision": precision,
        "recall": recall,
        "f1": compute_f1(precision, recall),
        "map50": to_float(metrics.box.map50),
        "map50_95": to_float(metrics.box.map),
        "speed_ms_per_image": speed,
    }


def efficiency_summary(model, imgsz):
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops

        flops_g = to_float(get_flops(model.model, imgsz=imgsz))
    except (ImportError, TypeError, RuntimeError, AttributeError):
        flops_g = None
    return {
        "params": parameter_count,
        "params_m": parameter_count / 1e6,
        "flops_g": flops_g,
    }


def runtime_versions():
    versions = {"python": platform.python_version()}
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["cuda"] = torch.version.cuda
    except ImportError:
        pass
    try:
        import ultralytics

        versions["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    return versions


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path
