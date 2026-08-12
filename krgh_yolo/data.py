"""Dataset validation and generated Ultralytics YAML files."""

from pathlib import Path


DATASET_PRESETS = {
    "scb": [
        "hand-raising",
        "reading",
        "writing",
        "using phone",
        "bowing the head",
        "leaning over the table",
    ],
    "scsb": [
        "head-down writing",
        "head-down reading",
        "head-up listening",
        "turning head",
        "hand-raising",
        "standing",
    ],
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_dataset_root(dataset_root):
    dataset_root = Path(dataset_root).expanduser().resolve()
    required = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "The dataset is missing required YOLO directories:\n"
            + "\n".join(missing)
        )
    return dataset_root


def count_dataset_files(dataset_root):
    dataset_root = Path(dataset_root)
    counts = {}
    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        counts[f"{split}_images"] = sum(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            for path in image_dir.rglob("*")
        )
        counts[f"{split}_labels"] = sum(
            path.is_file() and path.suffix.lower() == ".txt"
            for path in label_dir.rglob("*")
        )
    return counts


def _load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML with `pip install PyYAML`.") from exc
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def class_names_from_yaml(yaml_path):
    names = _load_yaml(yaml_path).get("names", [])
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda value: int(value))]
    if isinstance(names, list):
        return names
    return []


def write_dataset_yaml(dataset_root, output_dir, dataset_tag, class_names):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML with `pip install PyYAML`.") from exc

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"{dataset_tag}.yaml"
    config = {
        "path": str(Path(dataset_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return yaml_path


def resolve_data(data, output_dir, preset="scb", dataset_tag=None, class_names=None):
    """Resolve a dataset YAML or generate one from a YOLO directory root."""
    data = Path(data).expanduser()
    if data.suffix.lower() in {".yaml", ".yml"}:
        yaml_path = data.resolve()
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
        return {
            "yaml": yaml_path,
            "root": None,
            "class_names": class_names_from_yaml(yaml_path),
            "counts": None,
            "tag": dataset_tag or yaml_path.stem,
        }

    dataset_root = validate_dataset_root(data)
    if class_names:
        resolved_names = list(class_names)
    else:
        if preset not in DATASET_PRESETS:
            raise ValueError(
                "A directory dataset requires --preset scb/scsb or explicit "
                "--class-names."
            )
        resolved_names = DATASET_PRESETS[preset]
    tag = dataset_tag or preset
    yaml_path = write_dataset_yaml(dataset_root, output_dir, tag, resolved_names)
    return {
        "yaml": yaml_path.resolve(),
        "root": dataset_root,
        "class_names": resolved_names,
        "counts": count_dataset_files(dataset_root),
        "tag": tag,
    }
