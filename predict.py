"""Run KRGH-YOLO inference on images, videos, or directories."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run KRGH-YOLO inference.")
    parser.add_argument("weights", type=Path)
    parser.add_argument("source", help="Image, video, directory, URL, or webcam index.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/predict"))
    parser.add_argument("--name", default="krgh-yolo")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from krgh_yolo.checkpoint import load_yolo

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    model = load_yolo(weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        show=args.show,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=str(args.project.expanduser().resolve()),
        name=args.name,
    )
    if results:
        print(f"Predictions saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
