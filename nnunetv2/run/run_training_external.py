from __future__ import annotations

import argparse
import multiprocessing

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.run.run_training import run_training
from nnunetv2.utilities.external_dataset import materialize_external_dataset, resolve_dataset_name_and_id


def _ensure_paths_are_set() -> None:
    if not (nnUNet_raw and nnUNet_preprocessed and nnUNet_results):
        raise RuntimeError(
            "nnUNet_raw, nnUNet_preprocessed, and nnUNet_results must be set first."
        )


def _default_preprocess_processes(configuration: str) -> int:
    if configuration == "2d":
        return 8
    if configuration == "3d_fullres":
        return 4
    if configuration == "3d_lowres":
        return 8
    return 4


def _parse_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
        return torch.device("cpu")
    if device_name == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return torch.device("cuda")
    if device_name == "mps":
        return torch.device("mps")
    raise ValueError(f"-device must be either cpu, mps or cuda. Got: {device_name}.")


def run_external_training(args: argparse.Namespace) -> None:
    _ensure_paths_are_set()

    dataset_json = load_json(args.dataset_json)
    inferred_dataset_name = dataset_json.get("name")
    if args.dataset.startswith("Dataset"):
        dataset_id, dataset_name = resolve_dataset_name_and_id(args.dataset, args.dataset_name)
    else:
        dataset_id = int(args.dataset)
        dataset_name = args.dataset_name or inferred_dataset_name or f"External_{dataset_id:03d}"
        dataset_id, dataset_name = resolve_dataset_name_and_id(str(dataset_id), dataset_name)
    materialize_external_dataset(
        dataset_name=dataset_name,
        dataset_json_path=args.dataset_json,
        images_tr_dir=args.imagesTr,
        labels_tr_dir=args.labelsTr,
        images_ts_dir=args.imagesTs,
        stage_mode=args.stage_mode,
    )

    extract_fingerprints(
        [dataset_id],
        check_dataset_integrity=True,
        clean=args.clean,
        verbose=args.verbose,
        show_progress_bar=not args.no_pbar,
    )

    plans_identifier = args.p
    plan_experiments([dataset_id], overwrite_plans_name=plans_identifier)

    preprocess(
        [dataset_id],
        plans_identifier=plans_identifier,
        configurations=[args.configuration],
        num_processes=[_default_preprocess_processes(args.configuration)],
        verbose=args.verbose,
        show_progress_bar=not args.no_pbar,
    )

    device = _parse_device(args.device)
    run_training(
        str(dataset_id),
        args.configuration,
        args.fold,
        args.tr,
        plans_identifier,
        args.pretrained_weights,
        args.num_gpus,
        args.npz,
        args.c,
        args.val,
        args.disable_checkpointing,
        args.val_best,
        device=device,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset id or full dataset name")
    parser.add_argument("configuration", type=str, help="Configuration that should be trained")
    parser.add_argument("fold", type=str, help="Fold of the 5-fold cross-validation")
    parser.add_argument("--dataset_name", type=str, required=False, default=None,
                        help="Optional override for DatasetXXX_Name. Defaults to dataset_json['name'] if present, otherwise External_XXX.")
    parser.add_argument("--dataset_json", type=str, required=True,
                        help="Path to the source dataset.json file")
    parser.add_argument("--imagesTr", type=str, required=True,
                        help="Path to the training images folder")
    parser.add_argument("--labelsTr", type=str, required=True,
                        help="Path to the training labels folder")
    parser.add_argument("--imagesTs", type=str, required=False, default=None,
                        help="Optional path to the test images folder")
    parser.add_argument("--stage_mode", type=str, default="copy", choices=["copy", "symlink"],
                        help="How to stage imagesTs into nnUNet_raw")
    parser.add_argument("-tr", type=str, required=False, default="nnUNetTrainer",
                        help="[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer")
    parser.add_argument("-p", type=str, required=False, default="nnUNetPlans",
                        help="[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans")
    parser.add_argument("-pretrained_weights", type=str, required=False, default=None,
                        help="[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model")
    parser.add_argument("-num_gpus", type=int, default=1, required=False,
                        help="Specify the number of GPUs to use for training")
    parser.add_argument("--npz", action="store_true", required=False,
                        help="[OPTIONAL] Save softmax predictions from final validation as npz files")
    parser.add_argument("--c", action="store_true", required=False,
                        help="[OPTIONAL] Continue training from latest checkpoint")
    parser.add_argument("--val", action="store_true", required=False,
                        help="[OPTIONAL] Set this flag to only run the validation")
    parser.add_argument("--val_best", action="store_true", required=False,
                        help="[OPTIONAL] If set, validation will use checkpoint_best instead of checkpoint_final")
    parser.add_argument("--disable_checkpointing", action="store_true", required=False,
                        help="[OPTIONAL] Disable checkpointing")
    parser.add_argument("--clean", action="store_true", required=False,
                        help="[OPTIONAL] Overwrite existing fingerprints before preprocessing")
    parser.add_argument("--verbose", required=False, action="store_true",
                        help="Set this to print a lot of stuff. Useful for debugging.")
    parser.add_argument("--no_pbar", required=False, action="store_true",
                        help="Disable the progress bar. Recommended for cluster/HPC environments.")
    parser.add_argument("-device", type=str, default="cuda", required=False,
                        help="Use this to set the device the training should run with. Available options are 'cuda', 'cpu' and 'mps'.")
    return parser


def run_training_external_entry(argv: list[str] | None = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.device not in ["cpu", "cuda", "mps"]:
        raise ValueError(f"-device must be either cpu, mps or cuda. Got: {args.device}.")

    run_external_training(args)


if __name__ == "__main__":
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    run_training_external_entry()
