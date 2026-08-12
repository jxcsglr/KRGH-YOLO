"""Ultralytics trainer factory for KRGFusion and HHCA injection."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KRGHConfig:
    fusion: str = "krgf"
    head: str = "hhca"
    fusion_layers: tuple = (12, 15)
    fusion_num_layers: int = 2
    fusion_reduction: int = 4
    fusion_init_scale: float = 0.01
    fusion_strip_kernel: int = 7
    fusion_context: str = "local_strip"
    head_scales: tuple = (0, 1)
    head_num_scales: int = 2
    head_reduction: int = 4
    head_init_scale: float = 0.01
    head_strip_kernel: int = 7
    head_branch: str = "dual"
    injected_fusion: list = field(default_factory=list, compare=False)
    injected_head: list = field(default_factory=list, compare=False)


def inject_krgh_modules(model, config):
    """Inject the requested modules into an Ultralytics DetectionModel."""
    from krgh_yolo.hhca import (
        inject_classification_cbam_attention,
        inject_classification_head_hand_attention,
    )
    from krgh_yolo.krgfusion import inject_key_region_guided_fusion

    fusion_records = []
    head_records = []
    if config.fusion == "krgf":
        fusion_records = inject_key_region_guided_fusion(
            model,
            target_layers=list(config.fusion_layers) or None,
            num_fusions=config.fusion_num_layers,
            reduction=config.fusion_reduction,
            init_scale=config.fusion_init_scale,
            strip_kernel=config.fusion_strip_kernel,
            context_mode=config.fusion_context,
        )
    elif config.fusion != "none":
        raise ValueError(f"Unsupported fusion module: {config.fusion}")

    if config.head == "hhca":
        head_records = inject_classification_head_hand_attention(
            model,
            target_scales=list(config.head_scales) or None,
            num_scales=config.head_num_scales,
            reduction=config.head_reduction,
            init_scale=config.head_init_scale,
            strip_kernel=config.head_strip_kernel,
            branch_mode=config.head_branch,
        )
    elif config.head == "cbam":
        head_records = inject_classification_cbam_attention(
            model,
            target_scales=list(config.head_scales) or None,
            num_scales=config.head_num_scales,
            reduction=config.head_reduction,
            init_scale=config.head_init_scale,
        )
    elif config.head != "none":
        raise ValueError(f"Unsupported classification module: {config.head}")

    config.injected_fusion.clear()
    config.injected_fusion.extend(fusion_records)
    config.injected_head.clear()
    config.injected_head.extend(head_records)
    return fusion_records, head_records


def build_detection_trainer(config):
    """Create a DetectionTrainer subclass that injects modules after model build."""
    if config.fusion == "none" and config.head == "none":
        return None

    from ultralytics.models.yolo.detect import DetectionTrainer

    class KRGHDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            inject_krgh_modules(model, config)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            print(f"KRGH-YOLO parameters after injection: {parameter_count:,}")
            return model

    return KRGHDetectionTrainer
