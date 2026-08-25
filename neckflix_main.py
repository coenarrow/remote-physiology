""" The main function of rPPG deep learning pipeline.

Usage:
    Single GPU:
        uv run neckflix_main.py --config_file <path_to_config_file> --test_participants <list_of_test_participants>

    Single-Node Multi-GPU:
        uv run python -m torch.distributed.run --nproc_per_node=<num_gpus> neckflix_main.py --config_file <path_to_config_file> --test_participants <list_of_test_participants>

    Multi-Node Multi-GPU:
        Master Node (node_rank=0):
            uv run python -m torch.distributed.run --nnodes=<total_nodes> --node_rank=0 --nproc_per_node=<gpus_per_node> --rdzv-backend=c10d --rdzv-endpoint=<MASTER_IP>:29500 neckflix_main.py --config_file <path_to_config_file> --test_participants <list_of_test_participants>

        Worker Nodes (node_rank=1, 2, ...):
            uv run python -m torch.distributed.run --nnodes=<total_nodes> --node_rank=<N> --nproc_per_node=<gpus_per_node> --rdzv-backend=c10d --rdzv-endpoint=<MASTER_IP>:29500 neckflix_main.py --config_file <path_to_config_file> --test_participants <list_of_test_participants>

Examples:
    Single GPU:
        uv run neckflix_main.py --config_file physhydra_configs/physHydra_RGB_ABP.yaml --test_participants P015

    Single-Node with 4 GPUs:
        uv run python -m torch.distributed.run --nproc_per_node=4 neckflix_main.py --config_file physhydra_configs/physHydra_RGB_ABP.yaml --test_participants P015

    Multi-Node: 3 nodes × 2 GPUs each = 6 total GPUs:
        Node 0 (Master at 192.168.1.100):
            uv run python -m torch.distributed.run --nnodes=3 --node_rank=0 --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=192.168.1.100:29500 neckflix_main.py --config_file physhydra_configs/physHydra_RGB_ABP.yaml --test_participants P015

        Node 1 (Worker):
            uv run python -m torch.distributed.run --nnodes=3 --node_rank=1 --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=192.168.1.100:29500 neckflix_main.py --config_file physhydra_configs/physHydra_RGB_ABP.yaml --test_participants P015

        Node 2 (Worker):
            uv run python -m torch.distributed.run --nnodes=3 --node_rank=2 --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=192.168.1.100:29500 neckflix_main.py --config_file physhydra_configs/physHydra_RGB_ABP.yaml --test_participants P015

Notes:
    - For multi-node training, ensure all nodes can communicate on port 29500 (or your chosen port)
    - All nodes must have access to the same data and config files at identical paths
    - With N nodes and M GPUs per node, total world_size = N × M
    - Only rank 0 process saves models and outputs
"""

import argparse
import random
import time
import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader
import os
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


NUM_WORKERS = 4
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    parser.add_argument('--config_file', required=True, default="test_config_hydra.yaml", type=str, help="path to the config file")
    parser.add_argument('--test_participants', required=True, nargs='+', help="List of participants to test on")
    return parser

def train_and_test(config, data_loader_dict, rank=0, world_size=1):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == "FactorizePhys":
        model_trainer = trainer.FactorizePhysTrainer.FactorizePhysTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'PhysMamba':
        model_trainer = trainer.PhysMambaTrainer.PhysMambaTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'RhythmFormer':
        model_trainer = trainer.RhythmFormerTrainer.RhythmFormerTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'PhysHydra':
        model_trainer = trainer.PhysHydraTrainer.PhysHydraTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)

def test(config, data_loader_dict, rank=0, world_size=1):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)    
    elif config.MODEL.NAME == "FactorizePhys":
        model_trainer = trainer.FactorizePhysTrainer.FactorizePhysTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'PhysMamba':
        model_trainer = trainer.PhysMambaTrainer.PhysMambaTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'RhythmFormer':
        model_trainer = trainer.RhythmFormerTrainer.RhythmFormerTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    elif config.MODEL.NAME == 'PhysHydra':
        model_trainer = trainer.PhysHydraTrainer.PhysHydraTrainer(config, data_loader_dict, rank=rank, world_size=world_size,debug=config.DEBUG)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)

def unsupervised_method_inference(config, data_loader):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS")
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM")
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA")
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN")
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI")
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV")
        elif unsupervised_method == "OMIT":
            unsupervised_predict(config, data_loader, "OMIT")
        else:
            raise ValueError("Not supported unsupervised method!")

def main():
    """Main function for torchrun-based distributed training."""
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)

    # Determine if running with torchrun (distributed) or single process
    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if is_distributed:
        # Initialize process group when using torchrun
        setup_ddp()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set device for this process
        if 'cuda' in config.DEVICE:
            torch.cuda.set_device(local_rank)

        # Only print from rank 0 to avoid duplicate output
        is_main_process = (rank == 0)
    else:
        # Single process mode
        rank = 0
        world_size = 1
        local_rank = 0
        is_main_process = True

    if config.TRAIN.DATA.PREPROCESS.RESIZE.W == config.TEST.DATA.PREPROCESS.RESIZE.W:
        w = config.TRAIN.DATA.PREPROCESS.RESIZE.W
    else:
        raise ValueError("Train and test resize width must be the same!")

    if config.TRAIN.DATA.PREPROCESS.RESIZE.H == config.TEST.DATA.PREPROCESS.RESIZE.H:
        h = config.TRAIN.DATA.PREPROCESS.RESIZE.H
    else:
        raise ValueError("Train and test resize height must be the same!")

    if config.TRAIN.DATA.PREPROCESS.NECKFLIX.CHANNELS == config.TEST.DATA.PREPROCESS.NECKFLIX.CHANNELS:
        channels = ''.join(config.TRAIN.DATA.PREPROCESS.NECKFLIX.CHANNELS)
    else:
        raise ValueError("Train and test channels must be the same!")

    if config.TRAIN.DATA.PREPROCESS.NECKFLIX.TRACES == config.TEST.DATA.PREPROCESS.NECKFLIX.TRACES:
        traces = '-'.join(config.TRAIN.DATA.PREPROCESS.NECKFLIX.TRACES)
    else:
        raise ValueError("Train and test traces must be the same!")

    if config.TRAIN.DATA.PREPROCESS.NECKFLIX.POSTURES == config.TEST.DATA.PREPROCESS.NECKFLIX.POSTURES:
        postures = '-'.join(config.TRAIN.DATA.PREPROCESS.NECKFLIX.POSTURES)
    else:
        raise ValueError("Train and test postures must be the same!")

    # after defrost re-set the output save_dir, and the participant list
    config.defrost()
    test_participants_str = '_'.join(args.test_participants)
    exp_data_name = os.path.join(f"TRACES-{traces}_POSTURES-{postures}_CHANNELS-{channels}_H-{h}_W-{w}",f'tested_on_{test_participants_str}')
    config.TEST.OUTPUT_SAVE_DIR = config.TEST.OUTPUT_SAVE_DIR.replace(config.TEST.DATA.EXP_DATA_NAME, exp_data_name)
    config.TRAIN.DATA.EXP_DATA_NAME = exp_data_name
    config.TEST.DATA.EXP_DATA_NAME = exp_data_name
    config.MODEL.PHYSHYDRA.NUM_CHANNELS = len(config.TRAIN.DATA.PREPROCESS.NECKFLIX.CHANNELS)
    config.MODEL.PHYSHYDRA.NUM_LABELS = len(config.TRAIN.DATA.PREPROCESS.NECKFLIX.TRACES)
    new_save_dir = os.path.join(config.LOG.PATH, exp_data_name)
    config.TEST.OUTPUT_SAVE_DIR = new_save_dir
    config.MODEL.MODEL_DIR = os.path.join(config.LOG.PATH, config.TRAIN.DATA.EXP_DATA_NAME, "PreTrainedModels")
    config.freeze()

    # print out how many workers are used (only from main process)
    if is_main_process:
        print(f"Number of workers for data loading: {NUM_WORKERS}")
        if is_distributed:
            print(f"Running distributed training with {world_size} GPUs (Rank {rank}/{world_size})")
        else:
            print("Running in single process mode")

    # Check GPU availability (only from main process)
    if is_main_process and 'cuda' in config.DEVICE:
        num_cuda_devices = torch.cuda.device_count()
        if is_distributed:
            print(f"Each process will use 1 GPU. Total GPUs available: {num_cuda_devices}")
        else:
            print(f"Using CUDA with {num_cuda_devices} GPU(s) available")

    # Debug mode: set CUDA to synchronous for better error tracking
    if config.DEBUG:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.autograd.set_detect_anomaly(True)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        if is_main_process:
            print("DEBUG MODE: Running with synchronous CUDA and anomaly detection")

    # Set up the datasets (each rank creates its own)
    loader_dict = {'test': None, 'train': None, 'valid': None}
    test_loader = data_loader.NeckflixLoader.NeckflixLoader
    loader_dict['test'] = test_loader(name="test", data_path=config.TEST.DATA.DATA_PATH, config_data=config.TEST.DATA, device=config.DEVICE, test_participants=args.test_participants, get_raw_resized=config.MODEL.PHYSHYDRA.SAVE_ATTENTION_MAPS)
    if len(loader_dict['test']) == 0:
        raise ValueError("The test dataset is empty. Please check the test participants and data path.")
    train_loader = data_loader.NeckflixLoader.NeckflixLoader
    loader_dict['train'] = train_loader(name="train", data_path=config.TRAIN.DATA.DATA_PATH, config_data=config.TRAIN.DATA, device=config.DEVICE, test_participants=args.test_participants, get_raw_resized=False)
    valid_loader = data_loader.NeckflixLoader.NeckflixLoader
    loader_dict['valid'] = valid_loader(name="valid", data_path=config.TRAIN.DATA.DATA_PATH, config_data=config.TRAIN.DATA, device=config.DEVICE, test_participants=args.test_participants, get_raw_resized=False)

    if is_main_process:
        for split, dataset in loader_dict.items():
            print(f"{split} dataset has {len(dataset)} samples.")

    # create the output directory where test normally saves, and save a copy of the config there (only main process)
    if is_main_process:
        output_dir = config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # use os to copy the file to the output directory
        os.system(f"cp {args.config_file} {output_dir}/config.yaml")

    # Synchronize all processes before creating dataloaders
    if is_distributed:
        dist.barrier()

    # Create DataLoaders based on mode
    data_loader_dict = {'train': None, 'valid': None, 'test': None}
    assert 'test' in loader_dict
    assert 'train' in loader_dict
    assert 'valid' in loader_dict

    if config.TOOLBOX_MODE == "train_and_test":
        for split in ['train', 'valid']:
            if (split == 'valid' and config.TEST.USE_LAST_EPOCH):
                # skip validation loader creation if using last epoch
                data_loader_dict[split] = None
                continue
            if loader_dict[split] is None:
                data_loader_dict[split] = None
                continue
            # set up the sampler first
            if is_distributed and world_size > 1:
                sampler = DistributedSampler(loader_dict[split], num_replicas=world_size, rank=rank, shuffle=True)
            else:
                sampler = None
            data_loader_dict[split] = DataLoader(
                dataset=loader_dict[split],
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                worker_init_fn=seed_worker,
                drop_last=True
            )

    data_loader_dict['test'] = DataLoader(
        dataset=loader_dict['test'],
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
        sampler=None,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        drop_last=False
    )

    # Execute appropriate mode
    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict, rank, world_size)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict, rank, world_size)
    else:
        if is_main_process:
            print("TOOLBOX_MODE only support train_and_test or only_test!")

    # Cleanup distributed process group
    if is_distributed:
        cleanup_ddp()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Cleanup if an exception occurs
        if dist.is_initialized():
            cleanup_ddp()
        raise e