#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

from nnunetv2.run.run_training_external import run_training_external_entry

DATASET = "Dataset101_ThyroidUS"
CONFIGURATION = "2d"
FOLD = "0"
DATASET_JSON = r"/path/to/dataset.json"
IMAGES_TR = r"/path/to/imagesTr"
LABELS_TR = r"/path/to/labelsTr"
IMAGES_TS = None
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
DEVICE = "cpu"


def build_argv() -> list[str]:
    argv = [
        DATASET,
        CONFIGURATION,
        FOLD,
        "--dataset_json", DATASET_JSON,
        "--imagesTr", IMAGES_TR,
        "--labelsTr", LABELS_TR,
        "--stage_mode", STAGE_MODE,
        "-tr", TRAINER,
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


def main() -> None:
    if not (os.environ.get("nnUNet_raw") and os.environ.get("nnUNet_preprocessed") and os.environ.get("nnUNet_results")):
        raise RuntimeError("nnUNet_raw, nnUNet_preprocessed, and nnUNet_results must be set first.")
    run_training_external_entry(build_argv())


if __name__ == "__main__":
    main()
