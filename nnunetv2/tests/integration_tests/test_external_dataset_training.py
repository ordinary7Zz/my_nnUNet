from pathlib import Path

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.run import run_training_external
from nnunetv2.utilities.external_dataset import materialize_external_dataset


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    sitk.WriteImage(image, str(path))


def _create_source_dataset(root: Path) -> tuple[Path, Path, Path, Path]:
    images_tr = root / "imagesTr"
    labels_tr = root / "labelsTr"
    images_ts = root / "imagesTs"
    dataset_json = root / "dataset.json"

    for case_idx in range(2):
        case_id = f"case{case_idx + 1:03d}"
        image = np.zeros((8, 8), dtype=np.uint8)
        image[2:6, 2:6] = (case_idx + 1) * 10
        label = np.zeros((8, 8), dtype=np.uint8)
        label[3:5, 3:5] = 1

        _write_png(images_tr / f"{case_id}_0000.png", image)
        _write_png(labels_tr / f"{case_id}.png", label)

    _write_png(images_ts / "case900_0000.png", np.full((8, 8), 7, dtype=np.uint8))

    dataset_json.write_text(
        """{
  \"channel_names\": {\"0\": \"US\"},
  \"labels\": {\"background\": 0, \"nodule\": 1},
  \"numTraining\": 2,
  \"file_ending\": ".png"
}
""",
        encoding="utf-8",
    )

    return dataset_json, images_tr, labels_tr, images_ts


def test_materialize_external_dataset_writes_dataset_json(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "preprocessed"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "results"))

    source_root = tmp_path / "source"
    dataset_json, images_tr, labels_tr, images_ts = _create_source_dataset(source_root)

    dataset_root = materialize_external_dataset(
        dataset_name="Dataset123_ThyroidUS",
        dataset_json_path=str(dataset_json),
        images_tr_dir=str(images_tr),
        labels_tr_dir=str(labels_tr),
        images_ts_dir=str(images_ts),
        stage_mode="copy",
    )

    saved = load_json(str(dataset_root / "dataset.json"))
    assert saved["numTraining"] == 2
    assert saved["dataset"]["case001"]["images"][0] == str((images_tr / "case001_0000.png").resolve())
    assert saved["dataset"]["case001"]["label"] == str((labels_tr / "case001.png").resolve())
    assert (dataset_root / "imagesTs" / "case900_0000.png").is_file()


def test_external_training_entry_wires_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_raw", str(tmp_path / "raw"))
    monkeypatch.setenv("nnUNet_preprocessed", str(tmp_path / "preprocessed"))
    monkeypatch.setenv("nnUNet_results", str(tmp_path / "results"))

    source_root = tmp_path / "source"
    dataset_json, images_tr, labels_tr, images_ts = _create_source_dataset(source_root)

    calls = []

    def fake_extract_fingerprints(*args, **kwargs):
        calls.append(("extract", args, kwargs))

    def fake_plan_experiments(*args, **kwargs):
        calls.append(("plan", args, kwargs))
        return "nnUNetPlans"

    def fake_preprocess(*args, **kwargs):
        calls.append(("preprocess", args, kwargs))

    def fake_run_training(*args, **kwargs):
        calls.append(("train", args, kwargs))

    monkeypatch.setattr(run_training_external, "extract_fingerprints", fake_extract_fingerprints)
    monkeypatch.setattr(run_training_external, "plan_experiments", fake_plan_experiments)
    monkeypatch.setattr(run_training_external, "preprocess", fake_preprocess)
    monkeypatch.setattr(run_training_external, "run_training", fake_run_training)

    run_training_external.run_training_external_entry([
        "123",
        "2d",
        "0",
        "--dataset_name",
        "ThyroidUS",
        "--dataset_json",
        str(dataset_json),
        "--imagesTr",
        str(images_tr),
        "--labelsTr",
        str(labels_tr),
        "--imagesTs",
        str(images_ts),
        "--stage_mode",
        "copy",
        "--device",
        "cpu",
        "--no_pbar",
    ])

    assert [name for name, *_ in calls] == ["extract", "plan", "preprocess", "train"]
    assert calls[0][1][0] == [123]
    assert calls[2][1][0] == [123]
    assert calls[3][1][0] == 123
    assert (tmp_path / "raw" / "Dataset123_ThyroidUS" / "dataset.json").is_file()
