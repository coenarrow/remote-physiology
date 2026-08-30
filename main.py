""" The main function of rPPG deep learning pipeline.

Requires torchrun (torch.distributed.run) as the launcher for both single
and multi-GPU execution.  torchrun --nproc_per_node=1 is equivalent to
single-GPU training but provides a uniform DDP environment.

Usage:
    Single GPU:
        torchrun --nproc_per_node=1 main.py --config_file <config>

    Multi-GPU:
        torchrun --nproc_per_node=4 main.py --config_file <config>

    Neckflix LOSO Cross-Validation:
        torchrun --nproc_per_node=4 main.py --config_file <config> --test_participants P001

Examples:
    torchrun --nproc_per_node=1 main.py --config_file configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
    torchrun --nproc_per_node=4 main.py --config_file physhydra_configs/physHydra_RGB_ABP.yaml --test_participants P015
"""

import argparse
import random
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict

NUM_WORKERS = 4
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------------
# Registry maps: add new models / datasets / unsupervised methods here
# ---------------------------------------------------------------------------
TRAINER_REGISTRY = {
    "Physnet": trainer.PhysnetTrainer.PhysnetTrainer,
    "iBVPNet": trainer.iBVPNetTrainer.iBVPNetTrainer,
    "FactorizePhys": trainer.FactorizePhysTrainer.FactorizePhysTrainer,
    "Tscan": trainer.TscanTrainer.TscanTrainer,
    "EfficientPhys": trainer.EfficientPhysTrainer.EfficientPhysTrainer,
    "DeepPhys": trainer.DeepPhysTrainer.DeepPhysTrainer,
    "BigSmall": trainer.BigSmallTrainer.BigSmallTrainer,
    "PhysFormer": trainer.PhysFormerTrainer.PhysFormerTrainer,
    "PhysMamba": trainer.PhysMambaTrainer.PhysMambaTrainer,
    "RhythmFormer": trainer.RhythmFormerTrainer.RhythmFormerTrainer,
    "PhysHydra": trainer.PhysHydraTrainer.PhysHydraTrainer,
}

LOADER_REGISTRY = {
    "UBFC-rPPG": data_loader.UBFCrPPGLoader.UBFCrPPGLoader,
    "PURE": data_loader.PURELoader.PURELoader,
    "SCAMPS": data_loader.SCAMPSLoader.SCAMPSLoader,
    "MMPD": data_loader.MMPDLoader.MMPDLoader,
    "BP4DPlus": data_loader.BP4DPlusLoader.BP4DPlusLoader,
    "BP4DPlusBigSmall": data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader,
    "UBFC-PHYS": data_loader.UBFCPHYSLoader.UBFCPHYSLoader,
    "iBVP": data_loader.iBVPLoader.iBVPLoader,
    "PhysDrive": data_loader.PhysDriveLoader.PhysDriveLoader,
    "LADH": data_loader.LADHLoader.LADHLoader,
    "SUMS": data_loader.SUMSLoader.SUMSLoader,
}

UNSUPERVISED_METHODS = {"POS", "CHROM", "ICA", "GREEN", "LGI", "PBV", "OMIT"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_ddp():
    """Initialize distributed process group using environment variables set by torchrun."""
    init_process_group(backend="nccl")


def cleanup_ddp():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        destroy_process_group()


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml",
                        type=str, help="The name of the model.")
    parser.add_argument('--test_participants', required=False, nargs='+', default=None,
                        help="List of participants to test on (for LOSO cross-validation)")
    return parser


def get_loader_class(dataset_name):
    """Get the appropriate data loader class for a dataset."""
    if dataset_name not in LOADER_REGISTRY:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Currently supporting: {', '.join(LOADER_REGISTRY.keys())}"
        )
    return LOADER_REGISTRY[dataset_name]


def create_dataset(loader_class, name, data_path, config_data, device,
                   test_participants=None, get_raw_resized=False):
    """Create dataset instance with appropriate parameters."""
    kwargs = {
        "name": name,
        "data_path": data_path,
        "config_data": config_data,
        "device": device,
    }
    return loader_class(**kwargs)


# ---------------------------------------------------------------------------
# Train / test / unsupervised entry points
# ---------------------------------------------------------------------------
def train_and_test(config, data_loader_dict, rank, world_size):
    """Trains and then tests the model."""
    debug = getattr(config, 'DEBUG', False)
    trainer_cls = TRAINER_REGISTRY.get(config.MODEL.NAME)
    if trainer_cls is None:
        raise ValueError(f'Model {config.MODEL.NAME} is not supported. '
                         f'Available: {", ".join(TRAINER_REGISTRY.keys())}')
    model_trainer = trainer_cls(config, data_loader_dict,
                                rank=rank, world_size=world_size, debug=debug)
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict, rank, world_size):
    """Tests the model."""
    debug = getattr(config, 'DEBUG', False)
    trainer_cls = TRAINER_REGISTRY.get(config.MODEL.NAME)
    if trainer_cls is None:
        raise ValueError(f'Model {config.MODEL.NAME} is not supported. '
                         f'Available: {", ".join(TRAINER_REGISTRY.keys())}')
    model_trainer = trainer_cls(config, data_loader_dict,
                                rank=rank, world_size=world_size, debug=debug)
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader_dict):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for method in config.UNSUPERVISED.METHOD:
        if method not in UNSUPERVISED_METHODS:
            raise ValueError(f"Unsupported unsupervised method: {method}. "
                             f"Available: {', '.join(sorted(UNSUPERVISED_METHODS))}")
        unsupervised_predict(config, data_loader_dict, method)


# ---------------------------------------------------------------------------
# Neckflix-specific configuration
# ---------------------------------------------------------------------------
def apply_neckflix_specific_config(config, args):
    """
    Apply Neckflix-specific configuration logic.
    NOTE: This is Neckflix-specific logic that could be generalized in the future
    for other datasets that require similar LOSO cross-validation or dynamic naming.
    """
    if config.TRAIN.DATA.DATASET != "Neckflix" and config.TEST.DATA.DATASET != "Neckflix":
        return config

    # Validate consistency between train and test configs
    if config.TRAIN.DATA.PREPROCESS.RESIZE.W != config.TEST.DATA.PREPROCESS.RESIZE.W:
        raise ValueError("Train and test resize width must be the same!")
    if config.TRAIN.DATA.PREPROCESS.RESIZE.H != config.TEST.DATA.PREPROCESS.RESIZE.H:
        raise ValueError("Train and test resize height must be the same!")

    w = config.TRAIN.DATA.PREPROCESS.RESIZE.W
    h = config.TRAIN.DATA.PREPROCESS.RESIZE.H

    # Neckflix-specific: validate channels, traces, postures
    if hasattr(config.TRAIN.DATA.PREPROCESS, 'NECKFLIX') and hasattr(config.TEST.DATA.PREPROCESS, 'NECKFLIX'):
        if config.TRAIN.DATA.PREPROCESS.NECKFLIX.CHANNELS != config.TEST.DATA.PREPROCESS.NECKFLIX.CHANNELS:
            raise ValueError("Train and test channels must be the same!")
        if config.TRAIN.DATA.PREPROCESS.NECKFLIX.TRACES != config.TEST.DATA.PREPROCESS.NECKFLIX.TRACES:
            raise ValueError("Train and test traces must be the same!")
        if config.TRAIN.DATA.PREPROCESS.NECKFLIX.POSTURES != config.TEST.DATA.PREPROCESS.NECKFLIX.POSTURES:
            raise ValueError("Train and test postures must be the same!")

        channels = ''.join(config.TRAIN.DATA.PREPROCESS.NECKFLIX.CHANNELS)
        traces = '-'.join(config.TRAIN.DATA.PREPROCESS.NECKFLIX.TRACES)
        postures = '-'.join(config.TRAIN.DATA.PREPROCESS.NECKFLIX.POSTURES)

        # Construct experiment name dynamically
        config.defrost()
        if args.test_participants:
            test_participants_str = '_'.join(args.test_participants)
            exp_data_name = os.path.join(
                f"TRACES-{traces}_POSTURES-{postures}_CHANNELS-{channels}_H-{h}_W-{w}",
                f'tested_on_{test_participants_str}'
            )
        else:
            exp_data_name = f"TRACES-{traces}_POSTURES-{postures}_CHANNELS-{channels}_H-{h}_W-{w}"

        config.TEST.OUTPUT_SAVE_DIR = config.TEST.OUTPUT_SAVE_DIR.replace(
            config.TEST.DATA.EXP_DATA_NAME, exp_data_name
        )
        config.TRAIN.DATA.EXP_DATA_NAME = exp_data_name
        config.TEST.DATA.EXP_DATA_NAME = exp_data_name

        # Set model parameters from config (PhysHydra-specific, but could be generalized)
        if hasattr(config.MODEL, 'PHYSHYDRA'):
            config.MODEL.PHYSHYDRA.NUM_CHANNELS = len(config.TRAIN.DATA.PREPROCESS.NECKFLIX.CHANNELS)
            config.MODEL.PHYSHYDRA.NUM_LABELS = len(config.TRAIN.DATA.PREPROCESS.NECKFLIX.TRACES)

        new_save_dir = os.path.join(config.LOG.PATH, exp_data_name)
        config.TEST.OUTPUT_SAVE_DIR = new_save_dir
        config.MODEL.MODEL_DIR = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, "PreTrainedModels")
        config.freeze()

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Main function.  Always launched via torchrun."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # Get configuration
    config = get_config(args)

    # Initialize DDP (torchrun sets RANK, WORLD_SIZE, LOCAL_RANK)
    setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if 'cuda' in config.DEVICE:
        torch.cuda.set_device(local_rank)

    is_main_process = (rank == 0)

    # Apply dataset-specific configuration logic (e.g., Neckflix LOSO)
    config = apply_neckflix_specific_config(config, args)

    # Print setup information (only from rank 0)
    if is_main_process:
        print(f"Number of workers for data loading: {NUM_WORKERS}")
        print(f"Running with {world_size} GPU(s) (Rank {rank})")
        if 'cuda' in config.DEVICE:
            print(f"Total CUDA devices visible: {torch.cuda.device_count()}")

    # Debug mode
    if getattr(config, 'DEBUG', False):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.autograd.set_detect_anomaly(True)
        if is_main_process:
            print("DEBUG MODE: Running with synchronous CUDA and anomaly detection")

    # Create output directory and save config (only rank 0)
    if is_main_process:
        output_dir = config.TEST.OUTPUT_SAVE_DIR
        os.makedirs(output_dir, exist_ok=True)
        os.system(f"cp {args.config_file} {output_dir}/config.yaml")
        print(f"Saved a copy of the config file to: {output_dir}/config.yaml", end='\n\n')

    # Synchronize before creating dataloaders
    dist.barrier()

    # ------------------------------------------------------------------
    # Build data loaders
    # ------------------------------------------------------------------
    data_loader_dict = dict()

    if config.TOOLBOX_MODE == "train_and_test":
        # Training dataloader
        if config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH:
            train_dataset = create_dataset(
                get_loader_class(config.TRAIN.DATA.DATASET),
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                device=config.DEVICE,
                test_participants=args.test_participants,
            )
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                               rank=rank, shuffle=True)
            data_loader_dict['train'] = DataLoader(
                dataset=train_dataset,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=False,
                sampler=train_sampler,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=False,
                drop_last=True,
            )
        else:
            data_loader_dict['train'] = None

        # Validation dataloader
        if config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH:
            valid_dataset = create_dataset(
                get_loader_class(config.VALID.DATA.DATASET),
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA,
                device=config.DEVICE,
                test_participants=args.test_participants,
            )
            valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size,
                                               rank=rank, shuffle=False)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_dataset,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=False,
                sampler=valid_sampler,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=False,
                drop_last=True,
            )
        elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
            raise ValueError("Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        else:
            data_loader_dict['valid'] = None

    # Test dataloader (for train_and_test and only_test modes)
    if config.TOOLBOX_MODE in ("train_and_test", "only_test"):
        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            if is_main_process:
                print("Testing uses last epoch, validation dataset is not required.", end='\n\n')

        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            get_raw_resized = False
            if hasattr(config.MODEL, 'PHYSHYDRA') and hasattr(config.MODEL.PHYSHYDRA, 'SAVE_ATTENTION_MAPS'):
                get_raw_resized = config.MODEL.PHYSHYDRA.SAVE_ATTENTION_MAPS

            test_dataset = create_dataset(
                get_loader_class(config.TEST.DATA.DATASET),
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA,
                device=config.DEVICE,
                test_participants=args.test_participants,
                get_raw_resized=get_raw_resized,
            )
            if len(test_dataset) == 0:
                raise ValueError("The test dataset is empty. Please check the test participants and data path.")

            data_loader_dict["test"] = DataLoader(
                dataset=test_dataset,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                sampler=None,
                num_workers=NUM_WORKERS,
                pin_memory=False,
                worker_init_fn=seed_worker,
                persistent_workers=False,
                drop_last=False,
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_dataset = create_dataset(
            get_loader_class(config.UNSUPERVISED.DATA.DATASET),
            name="unsupervised",
            data_path=config.UNSUPERVISED.DATA.DATA_PATH,
            config_data=config.UNSUPERVISED.DATA,
            device=config.DEVICE,
        )
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_dataset,
            num_workers=NUM_WORKERS,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            pin_memory=False,
            persistent_workers=False,
        )
    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    # Print dataset sizes (rank 0 only)
    if is_main_process:
        for split, dl in data_loader_dict.items():
            if dl is not None:
                print(f"{split} dataset has {len(dl.dataset)} samples.")

    # Synchronize after dataloader creation
    dist.barrier()

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict, rank, world_size)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict, rank, world_size)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)

    cleanup_ddp()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        cleanup_ddp()
        raise e
