# partially borrowed from K-planes (https://github.com/sarafridov/K-Planes/blob/main/plenoxels/main.py)

import os
import argparse
import importlib.util
from typing import Any, Dict, List
import random
import numpy as np
from loguru import logger
import pprint

import torch

from datasets.synthetic_nerf_dataset import SyntheticNerfDataset
from models.ngp import NGPRadianceField
from runners.trainer import Trainer 


def init_dloader_random(_):
    seed = torch.initial_seed() % 2**32  # worker-specific seed initialized by pytorch
    np.random.seed(seed)
    random.seed(seed)
    
def get_model(config):
    if config['model_type'] == "NGP":
        logger.info(f"==> Initialize NGP model ...")
        aabb = [-1 * config["radius"]] * 3 + [config["radius"]] * 3
        model = NGPRadianceField(aabb=aabb).to('cuda')
    else:
        raise ValueError(f"Model type {config['model_type']} is invalid.")
    return model


def main():
    p = argparse.ArgumentParser(description="")

    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--config-path', type=str, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('override', nargs=argparse.REMAINDER)
    
    args = p.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(args.config_path), args.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config: Dict[str, Any] = cfg.config
    # Process overrides from argparse into config
    # overrides can be passed from the command line as key=value pairs. E.g.
    # python plenoxels/main.py --config-path plenoxels/config/cfg.py max_ts_frames=200
    # note that all values are strings, so code should assume incorrect data-types for anything
    # that's derived from config - and should not a string.
    overrides: List[str] = args.override
    overrides_dict = {ovr.split("=")[0]: ovr.split("=")[1] for ovr in overrides}
    config.update(overrides_dict)
    pprint.pprint(config)
    
    train_dataset = SyntheticNerfDataset(
        datadir = config["data_dir"],
        split="train",
        batch_size=config["batch_size"]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=4, prefetch_factor=4, pin_memory=True,
        batch_size=None, worker_init_fn=init_dloader_random)
    
    test_dataset = SyntheticNerfDataset(
        datadir = config["data_dir"],
        split="val",
    )
    
    model = get_model(config)
    trainer = Trainer(model, train_loader, train_dataset, test_dataset, **config)
    
    if args.validate_only:
        trainer.load_model()
        trainer.validate()
    else:
        trainer.train()
    
    

if __name__ == "__main__":
    main()
    