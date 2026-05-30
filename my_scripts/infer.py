#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

from nnunetv2.inference.predict_from_raw_data import predict_entry_point

INPUT_FOLDER = r"/path/to/input"
OUTPUT_FOLDER = r"/path/to/output"
DATASET = "Dataset101_ThyroidUS"
CONFIGURATION = "2d"
FOLDS = ["all"]
PLANS = "nnUNetPlans"
TRAINER = "nnUNetTrainer"
STEP_SIZE = 0.5
CHECKPOINT = "checkpoint_final.pth"
NPP = 3
NPS = 3
DEVICE = "cpu"
DISABLE_TTA = False
SAVE_PROBABILITIES = False
CONTINUE_PREDICTION = False
VERBOSE = False
DISABLE_PROGRESS_BAR = True
NOT_ON_DEVICE = False
PREV_STAGE_PREDICTIONS = None
NUM_PARTS = 1
PART_ID = 0


def build_argv() -> list[str]:
    argv = [
        "-i", INPUT_FOLDER,
        "-o", OUTPUT_FOLDER,
        "-d", DATASET,
        "-p", PLANS,
        "-tr", TRAINER,
        "-c", CONFIGURATION,
        "-f", *FOLDS,
        "-step_size", str(STEP_SIZE),
        "-chk", CHECKPOINT,
        "-npp", str(NPP),
        "-nps", str(NPS),
        "-num_parts", str(NUM_PARTS),
        "-part_id", str(PART_ID),
        "-device", DEVICE,
    ]
    if DISABLE_TTA:
        argv.append("--disable_tta")
    if SAVE_PROBABILITIES:
        argv.append("--save_probabilities")
    if CONTINUE_PREDICTION:
        argv.append("--continue_prediction")
    if VERBOSE:
        argv.append("--verbose")
    if DISABLE_PROGRESS_BAR:
        argv.append("--disable_progress_bar")
    if NOT_ON_DEVICE:
        argv.append("--not_on_device")
    if PREV_STAGE_PREDICTIONS:
        argv.extend(["-prev_stage_predictions", PREV_STAGE_PREDICTIONS])
    return argv


def main() -> None:
    if not (os.environ.get("nnUNet_raw") and os.environ.get("nnUNet_preprocessed") and os.environ.get("nnUNet_results")):
        raise RuntimeError("nnUNet_raw, nnUNet_preprocessed, and nnUNet_results must be set first.")
    sys.argv = ["nnUNetv2_predict", *build_argv()]
    predict_entry_point()


if __name__ == "__main__":
    main()
