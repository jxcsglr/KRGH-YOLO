"""Checkpoint loading with compatibility for the original research paths."""


def load_yolo(weights):
    # Import both current and historical module paths before torch unpickles a
    # checkpoint produced in the original research workspace.
    import krgh_yolo.hhca  # noqa: F401
    import krgh_yolo.krgfusion  # noqa: F401
    import experiments.ultralytics_baselines.key_region_fusion_modules  # noqa: F401
    import experiments.ultralytics_baselines.task_head_modules  # noqa: F401
    from ultralytics import YOLO

    return YOLO(str(weights))
