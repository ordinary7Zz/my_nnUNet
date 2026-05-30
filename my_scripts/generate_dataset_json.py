#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def file_ending(path: Path) -> str:
    return "".join(path.suffixes)


def strip_ending(name: str, ending: str) -> str:
    if ending and not name.endswith(ending):
        raise ValueError(f"{name} does not end with {ending}")
    return name[: -len(ending)] if ending else name


def list_files(folder: Path) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        raise ValueError(f"No files found in {folder}")
    return files


def infer_common_ending(files: list[Path]) -> str:
    ending = file_ending(files[0])
    if not ending:
        raise ValueError(f"Could not infer file ending from {files[0].name}")
    mismatched = [p.name for p in files if file_ending(p) != ending]
    if mismatched:
        raise ValueError(
            f"All files in a folder must share the same suffix. Expected {ending}, got examples: {mismatched[:5]}"
        )
    return ending


def build_pairs(images_dir: Path, labels_dir: Path) -> tuple[list[str], str, str]:
    image_files = list_files(images_dir)
    label_files = list_files(labels_dir)
    image_ending = infer_common_ending(image_files)
    label_ending = infer_common_ending(label_files)

    image_stems = {strip_ending(p.name, image_ending): p for p in image_files}
    label_stems = {strip_ending(p.name, label_ending): p for p in label_files}

    missing_labels = sorted(set(image_stems) - set(label_stems))
    missing_images = sorted(set(label_stems) - set(image_stems))
    if missing_labels or missing_images:
        raise ValueError(
            "Image/label filename stems do not match. "
            f"Missing labels: {missing_labels[:5]}; missing images: {missing_images[:5]}"
        )

    case_ids = sorted(image_stems)
    return case_ids, image_ending, label_ending


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an nnU-Net dataset.json from paired image/mask folders")
    parser.add_argument("dataset_root", type=Path, help="Root folder of your dataset")
    parser.add_argument("--images-subdir", type=str, default="images", help="Subfolder containing input images")
    parser.add_argument("--labels-subdir", type=str, default="masks", help="Subfolder containing binary masks")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write dataset.json. Defaults to <dataset_root>/dataset.json",
    )
    parser.add_argument(
        "--channel-name",
        type=str,
        default="image",
        help="Channel name used in nnU-Net normalization (use rgb_to_0_1 for RGB images)",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        default="foreground",
        help="Foreground class name for the binary mask",
    )
    parser.add_argument("--name", type=str, default=None, help="Dataset name to write into JSON")
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--reference", type=str, default=None)
    parser.add_argument("--release", type=str, default=None)
    parser.add_argument("--citation", type=str, default=None)
    parser.add_argument(
        "--overwrite-image-reader-writer",
        type=str,
        default=None,
        help="Optional nnU-Net reader/writer class name, e.g. NaturalImage2DIO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    images_dir = (dataset_root / args.images_subdir).resolve()
    labels_dir = (dataset_root / args.labels_subdir).resolve()

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels folder: {labels_dir}")

    case_ids, image_ending, label_ending = build_pairs(images_dir, labels_dir)
    output_json = args.output_json.resolve() if args.output_json else (dataset_root / "dataset.json").resolve()

    dataset_json = {
        "channel_names": {"0": args.channel_name},
        "labels": {"background": 0, args.label_name: 1},
        "numTraining": len(case_ids),
        "file_ending": image_ending,
        "name": args.name or dataset_root.name,
        "image_file_ending": image_ending,
        "label_file_ending": label_ending,
    }
    if args.description is not None:
        dataset_json["description"] = args.description
    if args.reference is not None:
        dataset_json["reference"] = args.reference
    if args.release is not None:
        dataset_json["release"] = args.release
    if args.citation is not None:
        dataset_json["citation"] = args.citation
    if args.overwrite_image_reader_writer is not None:
        dataset_json["overwrite_image_reader_writer"] = args.overwrite_image_reader_writer

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(dataset_json, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output_json}")
    if image_ending != label_ending:
        print(
            "Warning: image and label suffixes differ. nnU-Net's standard training folders still need a single "
            "common file_ending when you actually train. The JSON records both source endings for reference."
        )


if __name__ == "__main__":
    main()
