"""Validate a KRGH-YOLO checkpoint."""

import argparse
import json
from pathlib import Path

from krgh_yolo.data import DATASET_PRESETS, resolve_data


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a KRGH-YOLO checkpoint.")
    parser.add_argument("weights", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--preset", choices=sorted(DATASET_PRESETS), default="scb")
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/validation"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    project = args.project.expanduser().resolve()
    data_info = resolve_data(
        args.data,
        project / "_configs",
        preset=args.preset,
        dataset_tag=args.dataset_tag,
        class_names=args.class_names,
    )
    from krgh_yolo.checkpoint import load_yolo
    from krgh_yolo.reporting import efficiency_summary, metric_summary, write_json

    model = load_yolo(weights)
    name = args.name or f"{args.weights.stem}_{data_info['tag']}"
    metrics = model.val(
        data=str(data_info["yaml"]),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        split="val",
        plots=True,
        save_json=args.save_json,
        project=str(project),
        name=name,
    )
    output_dir = Path(getattr(metrics, "save_dir", project / name))
    summary = {
        "weights": str(weights),
        "dataset_yaml": str(data_info["yaml"]),
        "metrics": metric_summary(metrics),
        "efficiency": efficiency_summary(model, args.imgsz),
    }
    output = write_json(output_dir / "validation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary saved to: {output}")


if __name__ == "__main__":
    main()
