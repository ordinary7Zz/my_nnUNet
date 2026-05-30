from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple

from batchgenerators.utilities.file_and_folder_operations import load_json, maybe_mkdir_p, save_json

from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder

_DATASET_NAME_RE = re.compile(r"^Dataset(?P<dataset_id>\d{3})_(?P<dataset_name>.+)$")


def resolve_dataset_name_and_id(dataset: str, dataset_name: Optional[str] = None) -> Tuple[int, str]:
    if dataset.startswith("Dataset"):
        match = _DATASET_NAME_RE.fullmatch(dataset)
        if match is None:
            raise ValueError(
                f"dataset must match the pattern DatasetXXX_Name. Got: {dataset}"
            )
        return int(match.group("dataset_id")), dataset

    dataset_id = int(dataset)
    if dataset_name is None:
        raise ValueError(
            "dataset_name is required when dataset is numeric. Provide the suffix name or a full DatasetXXX_Name."
        )

    if dataset_name.startswith("Dataset"):
        match = _DATASET_NAME_RE.fullmatch(dataset_name)
        if match is None:
            raise ValueError(
                f"dataset_name must match the pattern DatasetXXX_Name. Got: {dataset_name}"
            )
        parsed_id = int(match.group("dataset_id"))
        if parsed_id != dataset_id:
            raise ValueError(
                f"dataset id mismatch: got {dataset_id}, but dataset_name implies {parsed_id}."
            )
        return dataset_id, dataset_name

    return dataset_id, f"Dataset{dataset_id:03d}_{dataset_name}"


def _build_training_mapping(images_tr_dir: Path, labels_tr_dir: Path, file_ending: str, num_channels: int):
    case_ids = get_identifiers_from_splitted_dataset_folder(str(images_tr_dir), file_ending)
    if len(case_ids) == 0:
        raise FileNotFoundError(
            f"No training images with suffix {file_ending} were found in {images_tr_dir}."
        )

    dataset = {}
    for case_id in case_ids:
        images = [
            (images_tr_dir / f"{case_id}_{channel_idx:04d}{file_ending}").resolve()
            for channel_idx in range(num_channels)
        ]
        label_file = (labels_tr_dir / f"{case_id}{file_ending}").resolve()

        missing_images = [str(p) for p in images if not p.is_file()]
        if missing_images:
            raise FileNotFoundError(
                f"Missing expected training image files for case {case_id}: {missing_images}"
            )
        if not label_file.is_file():
            raise FileNotFoundError(
                f"Missing expected label file for case {case_id}: {label_file}"
            )

        dataset[case_id] = {
            "images": [str(p) for p in images],
            "label": str(label_file),
        }

    return dataset


def _remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _stage_images_ts(images_ts_dir: Path, target_dir: Path, stage_mode: str) -> None:
    if target_dir.exists() or target_dir.is_symlink():
        _remove_existing_path(target_dir)

    maybe_mkdir_p(str(target_dir))

    for source_file in images_ts_dir.rglob("*"):
        if source_file.is_dir():
            continue

        relative_path = source_file.relative_to(images_ts_dir)
        destination = target_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        if stage_mode == "copy":
            shutil.copy2(source_file, destination)
        elif stage_mode == "symlink":
            destination.symlink_to(source_file.resolve())
        else:
            raise ValueError(f"Unknown stage_mode: {stage_mode}")


def materialize_external_dataset(
    dataset_name: str,
    dataset_json_path: str,
    images_tr_dir: str,
    labels_tr_dir: str,
    images_ts_dir: Optional[str] = None,
    stage_mode: str = "copy",
) -> Path:
    dataset_json = load_json(dataset_json_path)
    required_keys = ["channel_names", "labels", "file_ending"]
    missing_keys = [key for key in required_keys if key not in dataset_json]
    if missing_keys:
        raise ValueError(
            f"dataset_json is missing required keys: {missing_keys}"
        )

    images_tr_path = Path(images_tr_dir).resolve()
    labels_tr_path = Path(labels_tr_dir).resolve()
    file_ending = dataset_json["file_ending"]
    num_channels = len(dataset_json["channel_names"])
    dataset = _build_training_mapping(images_tr_path, labels_tr_path, file_ending, num_channels)

    dataset_root = Path(os.fspath(nnUNet_raw)) / dataset_name
    maybe_mkdir_p(str(dataset_root))

    dataset_json["dataset"] = dataset
    dataset_json["numTraining"] = len(dataset)
    save_json(dataset_json, str(dataset_root / "dataset.json"), sort_keys=False)

    if images_ts_dir is not None:
        _stage_images_ts(Path(images_ts_dir).resolve(), dataset_root / "imagesTs", stage_mode)

    return dataset_root
