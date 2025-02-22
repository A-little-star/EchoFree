import os
import yaml
import torch as th
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import numpy as np
import soundfile as sf
import argparse
from pathlib import Path
from tqdm import tqdm
import librosa
import concurrent.futures

from collections import OrderedDict

import sys,os
sys.path.append(
    os.path.dirname(__file__))
sys.path.append(os.path.dirname(sys.path[0]))

from libs.import_module import import_module

def load_obj(obj, device):
    '''
    Offload tensor object in obj to cuda device
    '''
    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj
    
    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)

def inference(task_id, data_reader, nnet, conf, device, mode, 
              ):
    with th.no_grad():
        print(f'{mode} decoding:')
        for egs in tqdm(data_reader):
            egs = load_obj(egs, device)
            egs["mix"] = egs["mix"].contiguous()
            egs["ref"] = egs["ref"].contiguous()
            egs['laec_out'] = egs['laec_out'].contiguous()
            egs['laec_echo'] = egs['laec_echo'].contiguous()

            inputs = egs['mix'].squeeze(0).detach().cpu()
            laec_out = egs['laec_out'].squeeze(0).detach().cpu()
            far = egs['ref'].squeeze(0).detach().cpu()
            laec_echo = egs['laec_echo'].squeeze(0).detach().cpu()

            inputs = inputs.to(device)
            laec_echo = laec_echo.to(device)

            outputs = nnet(inputs.unsqueeze(0), laec_echo.unsqueeze(0))
            outputs_wav = outputs["wavs"].squeeze(0)

            # outputs_wav = outputs_wav / torch.max(torch.abs(outputs_wav)) * torch.max(torch.abs(inputs))

            out = outputs_wav.detach().cpu().numpy()

            if not os.path.exists(os.path.join(conf["save"]["dir"], mode)):
                os.makedirs(os.path.join(conf["save"]["dir"], mode))
            save_path = os.path.join(conf["save"]["dir"], mode, egs["utt_id"])
            sf.write(save_path, out, conf["save"]["sample_rate"])

def run(args):
    torch.set_printoptions(profile="full")
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])    
    dist.init_process_group(args.backend, rank=WORLD_RANK, world_size=WORLD_SIZE)    
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda', LOCAL_RANK)
    print(f"[{os.getpid()}] using device: {device}", torch.cuda.current_device(), "local rank", LOCAL_RANK)    
    
    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    
    datareader_path = conf["test"]["datareader_path"]
    datareader_name = conf["test"]["datareader_name"]
    DataReader = import_module(datareader_path, datareader_name)

    conf["datareader"]["filename"] = conf["testlist"]["goer_list"]
    data_reader_goer = DataReader(**conf["datareader"])

    module_path = conf['test']['module_path']
    module_name = conf['test']['module_name']
    model = import_module(module_path, module_name)

    nnet = model(**conf["nnet_conf"])
    nnet = nnet.to(device)
    nnet = DistributedDataParallel(nnet, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True) 

    checkpoint_dir = Path(conf["test"]["checkpoint"])
    if os.path.isfile(checkpoint_dir):
        cpt_fname = checkpoint_dir
    elif os.path.isdir(checkpoint_dir):
        cpt_fname = checkpoint_dir / "best.pt.tar"
    # nnet.reload_spk()
    cpt = th.load(cpt_fname, map_location="cpu")

    nnet.load_state_dict(cpt["model_state_dict"], strict=False)
    nnet = nnet.to(device)
    nnet.eval()

    if not os.path.exists(conf["save"]["dir"]):
        os.makedirs(conf["save"]["dir"], exist_ok=True)  
    
    inference(3, data_reader_goer, nnet, conf, device, "goer")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Command to test separation model in Pytorch",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-conf",
                        type=str,
                        required=True,
                        help="Yaml configuration file for training")
    parser.add_argument("--backend",
                        type=str,
                        default="nccl",
                        choices=["nccl", "gloo"])
    args = parser.parse_args()

    # for nccl debug
    os.environ["NCCL_DEBUG"] = "INFO"    
    
    run(args)
    
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port=21547 inference.py -conf ./config/test.yml