#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

from nnunetv2.run.run_training_external import run_training_external_entry

DATASET = "Dataset101_ThyroidUS"
CONFIGURATION = "2d"
FOLD = "0"
DATASET_JSON = r"/mnt/wangbd8/workspace/ThyroidAgent/Segmentation_Models/my_nnUNet/dataset.json"
IMAGES_TR = r"/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_1/train/images"
LABELS_TR = r"/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_1/train/masks"
IMAGES_TS = r"/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_1/test/images"
LABELS_TS = r"/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_experiment/dataset_1/test/masks"
STAGE_MODE = "copy"
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetPlans"
PRETRAINED_WEIGHTS = None
NUM_GPUS = 1
SAVE_NPZ = False
CONTINUE_TRAINING = False
VALIDATE_ONLY = False
VAL_BEST = False
DISABLE_CHECKPOINTING = False
CLEAN = False
VERBOSE = False
NO_PBAR = True
DEVICE = "cuda:0"
EPOCHS = 10


def build_argv(trainer_name: str) -> list[str]:
    argv = [
        DATASET,
        CONFIGURATION,
        FOLD,
        "--dataset_json", DATASET_JSON,
        "--imagesTr", IMAGES_TR,
        "--labelsTr", LABELS_TR,
        "--stage_mode", STAGE_MODE,
        "-tr", trainer_name,
        "-p", PLANS,
        "-num_gpus", str(NUM_GPUS),
        "-device", DEVICE,
    ]
    if IMAGES_TS:
        argv.extend(["--imagesTs", IMAGES_TS])
    if PRETRAINED_WEIGHTS:
        argv.extend(["-pretrained_weights", PRETRAINED_WEIGHTS])
    if SAVE_NPZ:
        argv.append("--npz")
    if CONTINUE_TRAINING:
        argv.append("--c")
    if VALIDATE_ONLY:
        argv.append("--val")
    if VAL_BEST:
        argv.append("--val_best")
    if DISABLE_CHECKPOINTING:
        argv.append("--disable_checkpointing")
    if CLEAN:
        argv.append("--clean")
    if VERBOSE:
        argv.append("--verbose")
    if NO_PBAR:
        argv.append("--no_pbar")
    return argv


def _run_training(argv: list[str]) -> None:
    if EPOCHS is None:
        run_training_external_entry(argv)
        return

    trainer_class = f"nnUNetTrainer_{EPOCHS}epochs"
    trainer_code = dedent(
        f"""\
        import torch

        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


        class {trainer_class}(nnUNetTrainer):
            def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                         device: torch.device = torch.device('cuda')):
                super().__init__(plans, configuration, fold, dataset_json, device)
                self.num_epochs = {EPOCHS}
        """
    )

    previous_ext_trainer = os.environ.get("nnUNet_extTrainer")
    with TemporaryDirectory() as ext_trainer_dir:
        Path(ext_trainer_dir, f"epoch_trainer_{EPOCHS}.py").write_text(trainer_code, encoding="utf-8")
        os.environ["nnUNet_extTrainer"] = (
            ext_trainer_dir if previous_ext_trainer is None
            else os.pathsep.join([ext_trainer_dir, previous_ext_trainer])
        )
        try:
            run_training_external_entry(argv)
        finally:
            if previous_ext_trainer is None:
                os.environ.pop("nnUNet_extTrainer", None)
            else:
                os.environ["nnUNet_extTrainer"] = previous_ext_trainer


def main() -> None:
    if not (os.environ.get("nnUNet_raw") and os.environ.get("nnUNet_preprocessed") and os.environ.get("nnUNet_results")):
        raise RuntimeError("nnUNet_raw, nnUNet_preprocessed, and nnUNet_results must be set first.")
    trainer_name = TRAINER if EPOCHS is None else f"nnUNetTrainer_{EPOCHS}epochs"
    _run_training(build_argv(trainer_name))


if __name__ == "__main__":
    main()
