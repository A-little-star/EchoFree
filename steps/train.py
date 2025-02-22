import sys
import os
import time
import yaml
import pprint
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.quantization

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

sys.path.append(
    os.path.dirname(__file__))
sys.path.append(os.path.dirname(sys.path[0]))

# from loader.dataloader_pfdkf_fd_kiss_fft import make_loader

from libs.import_module import import_module
# from trainer.trainer import Trainer

def make_optimizer(params, opt):
    '''
    make optimizer
    '''
    supported_optimizer = {
        "sgd": torch.optim.SGD,  # momentum, weight_decay, lr
        "rmsprop": torch.optim.RMSprop,  # momentum, weight_decay, lr
        "adam": torch.optim.Adam,  # weight_decay, lr
        "adadelta": torch.optim.Adadelta,  # weight_decay, lr
        "adagrad": torch.optim.Adagrad,  # lr, lr_decay, weight_decay
        "adamax": torch.optim.Adamax,  # lr, weight
        "adamw": torch.optim.AdamW
        # ...
    }

    if opt['optim']['name'] not in supported_optimizer:
        raise ValueError("Now only support optimizer {}".format(opt['optim']['name']))
    optimizer = supported_optimizer[opt['optim']['name']](params, **opt['optim']['optimizer_kwargs'])
    return optimizer

def make_dataloader(opt, make_loader):
    if opt['test']:
        train_dataloader = make_loader(
            opt['datasets']['test']['data_conf'],
            **opt['datasets']['dataloader_setting'],
        )
        valid_dataloader = make_loader(
            opt['datasets']['test']['data_conf'],
            **opt['datasets']['dataloader_setting'],
        )
    else:
        train_dataloader = make_loader(
            opt['datasets']['train']['data_conf'],
            **opt['datasets']['dataloader_setting'],
        )
        valid_dataloader = make_loader(
            opt['datasets']['val']['data_conf'],
            **opt['datasets']['dataloader_setting'],
        )
    return train_dataloader, valid_dataloader


def main_worker(args):

    print("Arguments in args:\n{}".format(pprint.pformat(vars(args))), flush=True)
    
    # Environment variables set by torch.distributed.launch
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

    # initialize distributed training environment
    dist.init_process_group(args.backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

    # log environment variables
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")

    torch.cuda.set_device(LOCAL_RANK)    
    device = torch.device('cuda:{}'.format(int(LOCAL_RANK)))
    cudnn.benchmark = True

    if LOCAL_RANK == 0:
        print("Arguments in args:\n{}".format(pprint.pformat(vars(args))), flush=True)

    # load configurations
    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    if LOCAL_RANK == 0:        
        print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)

    checkpoint_dir = Path(conf['train']['checkpoint'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    random.seed(conf['train']['seed'])
    np.random.seed(conf['train']['seed'])
    torch.cuda.manual_seed_all(conf['train']['seed'])

    # if exist, resume training
    last_checkpoint = checkpoint_dir / "last.pt.tar"
    if last_checkpoint.exists() and LOCAL_RANK == 0:
        print(f"Found old checkpoint: {last_checkpoint}", flush=True)
        conf['train']['resume'] = last_checkpoint.as_posix()

    # dump configurations
    with open(checkpoint_dir / "train.yaml", "w") as f:
        yaml.dump(conf, f)
    
    #build nnet
    module_path = conf['train']['module_path']
    module_name = conf['train']['module_name']
    model = import_module(module_path, module_name)

    nnet = model(**conf["nnet_conf"]).to(device)
    nnet = DistributedDataParallel(nnet, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)    

    # build optimizer
    optimizer = make_optimizer(filter(lambda p: p.requires_grad, nnet.parameters()), conf)
    # build dataloader
    dataloader_path = conf['train']['dataloader_path']
    dataloader_name = conf['train']['dataloader_name']
    make_loader = import_module(dataloader_path, dataloader_name)
    train_loader, valid_loader = make_dataloader(conf, make_loader)
    # build scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=conf['scheduler']['factor'],
        patience=conf['scheduler']['patience'],
        min_lr=conf['scheduler']['min_lr'],
        verbose=True)
    
    trainer_path = conf['train']['trainer_path']
    trainer_name = conf['train']['trainer_name']
    Trainer = import_module(trainer_path, trainer_name)
    trainer = Trainer(nnet,
                      optimizer,
                      scheduler,
                      device,
                      conf,
                      LOCAL_RANK,
                      WORLD_SIZE)

    
    trainer.run(train_loader,
                valid_loader,
                num_epoches=conf['train']['epoch'],
                test=conf['test'],
               )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--local_rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
    parser.add_argument("--backend",
                        type=str,
                        default="nccl",
                        choices=["nccl", "gloo"])
    parser.add_argument('-conf',
                    type=str,
                    required=True,
                    help='Yaml configuration file for training')
    args = parser.parse_args()
    main_worker(args)