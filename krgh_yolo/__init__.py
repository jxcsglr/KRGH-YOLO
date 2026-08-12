"""KRGH-YOLO model components.

The heavy PyTorch modules are imported lazily so dataset validation and CLI help
remain available before the training dependencies are installed.
"""

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    "KeyRegionGuidedFusion": ("krgh_yolo.krgfusion", "KeyRegionGuidedFusion"),
    "KeyRegionFusionWrapper": ("krgh_yolo.krgfusion", "KeyRegionFusionWrapper"),
    "inject_key_region_guided_fusion": (
        "krgh_yolo.krgfusion",
        "inject_key_region_guided_fusion",
    ),
    "ClassificationHeadHandAttention": (
        "krgh_yolo.hhca",
        "ClassificationHeadHandAttention",
    ),
    "ClassificationCBAMAttention": (
        "krgh_yolo.hhca",
        "ClassificationCBAMAttention",
    ),
    "CoordinateGate": ("krgh_yolo.hhca", "CoordinateGate"),
    "inject_classification_head_hand_attention": (
        "krgh_yolo.hhca",
        "inject_classification_head_hand_attention",
    ),
    "inject_classification_cbam_attention": (
        "krgh_yolo.hhca",
        "inject_classification_cbam_attention",
    ),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
