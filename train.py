"""Train and evaluate KRGH-YOLO on a YOLO-format dataset."""

import argparse
import json
from pathlib import Path

from krgh_yolo.data import DATASET_PRESETS, resolve_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train KRGH-YOLO (YOLO11s + KRGFusion + HHCA)."
    )
    parser.add_argument(
        "data",
        type=Path,
        help="Dataset YAML or YOLO root containing images/ and labels/.",
    )
    parser.add_argument("--preset", choices=sorted(DATASET_PRESETS), default="scb")
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Ultralytics model source. Use yolo11s.yaml for scratch training.",
    )

    parser.add_argument("--fusion", choices=["none", "krgf"], default="krgf")
    parser.add_argument("--head", choices=["none", "hhca", "cbam"], default="hhca")
    parser.add_argument("--krgf-layers", nargs="*", type=int, default=[12, 15])
    parser.add_argument("--krgf-num-layers", type=int, default=2)
    parser.add_argument("--krgf-context", choices=["local", "strip", "local_strip"], default="local_strip")
    parser.add_argument("--krgf-reduction", type=int, default=4)
    parser.add_argument("--krgf-init-scale", type=float, default=0.01)
    parser.add_argument("--krgf-strip-kernel", type=int, default=7)
    parser.add_argument("--hhca-scales", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--hhca-num-scales", type=int, default=2)
    parser.add_argument("--hhca-branch", choices=["posture", "hand_object", "dual"], default="dual")
    parser.add_argument("--hhca-reduction", type=int, default=4)
    parser.add_argument("--hhca-init-scale", type=float, default=0.01)
    parser.add_argument("--hhca-strip-kernel", type=int, default=7)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--amp", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--project", type=Path, default=Path("runs/krgh-yolo"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_architecture_args(args):
    if args.krgf_strip_kernel <= 0 or args.krgf_strip_kernel % 2 == 0:
        raise ValueError("--krgf-strip-kernel must be a positive odd integer.")
    if args.hhca_strip_kernel <= 0 or args.hhca_strip_kernel % 2 == 0:
        raise ValueError("--hhca-strip-kernel must be a positive odd integer.")
    if args.krgf_reduction <= 0 or args.hhca_reduction <= 0:
        raise ValueError("Reduction ratios must be positive.")


def default_run_name(args, dataset_tag):
    modules = "_".join(part for part in (args.fusion, args.head) if part != "none")
    modules = modules or "baseline"
    return f"yolo11s_{modules}_{dataset_tag}_e{args.epochs}_img{args.imgsz}_seed{args.seed}"


def printable_config(args, data_info, run_name):
    return {
        "data": str(data_info["yaml"]),
        "dataset_root": str(data_info["root"]) if data_info["root"] else None,
        "dataset_tag": data_info["tag"],
        "class_names": data_info["class_names"],
        "file_counts": data_info["counts"],
        "model": args.model,
        "run_name": run_name,
        "project": str(args.project.resolve()),
        "fusion": args.fusion,
        "head": args.head,
        "krgf_layers": args.krgf_layers,
        "krgf_context": args.krgf_context,
        "hhca_scales": args.hhca_scales,
        "hhca_branch": args.hhca_branch,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
    }


def main():
    args = parse_args()
    validate_architecture_args(args)
    args.project = args.project.expanduser().resolve()
    data_info = resolve_data(
        args.data,
        args.project / "_configs",
        preset=args.preset,
        dataset_tag=args.dataset_tag,
        class_names=args.class_names,
    )
    run_name = args.name or default_run_name(args, data_info["tag"])
    config_for_print = printable_config(args, data_info, run_name)
    print(json.dumps(config_for_print, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run complete. Training was not started.")
        return

    from krgh_yolo.checkpoint import load_yolo
    from krgh_yolo.reporting import (
        efficiency_summary,
        metric_summary,
        runtime_versions,
        write_json,
    )
    from krgh_yolo.trainer import KRGHConfig, build_detection_trainer

    krgh_config = KRGHConfig(
        fusion=args.fusion,
        head=args.head,
        fusion_layers=tuple(args.krgf_layers),
        fusion_num_layers=args.krgf_num_layers,
        fusion_reduction=args.krgf_reduction,
        fusion_init_scale=args.krgf_init_scale,
        fusion_strip_kernel=args.krgf_strip_kernel,
        fusion_context=args.krgf_context,
        head_scales=tuple(args.hhca_scales),
        head_num_scales=args.hhca_num_scales,
        head_reduction=args.hhca_reduction,
        head_init_scale=args.hhca_init_scale,
        head_strip_kernel=args.hhca_strip_kernel,
        head_branch=args.hhca_branch,
    )
    trainer = build_detection_trainer(krgh_config)
    model = load_yolo(args.model)
    train_kwargs = {
        "data": str(data_info["yaml"]),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "cache": args.cache,
        "project": str(args.project),
        "name": run_name,
        "exist_ok": args.exist_ok,
        "plots": True,
        "save": True,
        "val": True,
    }
    if trainer is not None:
        train_kwargs["trainer"] = trainer
    if args.amp != "auto":
        train_kwargs["amp"] = args.amp == "true"
    model.train(**train_kwargs)

    run_dir = Path(model.trainer.save_dir).resolve()
    best_weights = Path(model.trainer.best).resolve()
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training finished without best.pt: {best_weights}")
    best_model = load_yolo(best_weights)
    metrics = best_model.val(
        data=str(data_info["yaml"]),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        split="val",
        plots=True,
        save_json=args.save_json,
        project=str(args.project),
        name=f"{run_dir.name}_best_val",
    )
    summary = {
        "run_name": run_dir.name,
        "model_source": args.model,
        "best_weights": str(best_weights),
        "dataset_yaml": str(data_info["yaml"]),
        "dataset_root": str(data_info["root"]) if data_info["root"] else None,
        "dataset_tag": data_info["tag"],
        "class_names": data_info["class_names"],
        "dataset_counts": data_info["counts"],
        "training": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "seed": args.seed,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "weight_decay": args.weight_decay,
        },
        "model": {
            "fusion": args.fusion,
            "head": args.head,
            "krgf_layers": args.krgf_layers,
            "krgf_context": args.krgf_context,
            "krgf_reduction": args.krgf_reduction,
            "krgf_init_scale": args.krgf_init_scale,
            "krgf_strip_kernel": args.krgf_strip_kernel,
            "hhca_scales": args.hhca_scales,
            "hhca_branch": args.hhca_branch,
            "hhca_reduction": args.hhca_reduction,
            "hhca_init_scale": args.hhca_init_scale,
            "hhca_strip_kernel": args.hhca_strip_kernel,
            "injected_fusion": krgh_config.injected_fusion,
            "injected_head": krgh_config.injected_head,
        },
        "checkpoint_selection": {
            "checkpoint": "best.pt",
            "criterion": "Ultralytics validation fitness",
            "formula": "0.1 * mAP50 + 0.9 * mAP50:95",
            "reported_metrics_source": "best.pt re-evaluated on validation",
        },
        "metrics": metric_summary(metrics),
        "efficiency": efficiency_summary(best_model, args.imgsz),
        "runtime": runtime_versions(),
    }
    summary_path = write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
