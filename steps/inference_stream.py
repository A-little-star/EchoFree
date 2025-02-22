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
import torch.nn.functional as F

from collections import OrderedDict

import sys,os
sys.path.append(
    os.path.dirname(__file__))
sys.path.append(os.path.dirname(sys.path[0]))

from libs.import_module import import_module

def vorbis_window(n):
    """Generate a Vorbis window of length n."""
    window = np.zeros(n)
    for i in range(n):
        window[i] = np.sin((np.pi / 2) * (np.sin(np.pi * (i + 0.5) / n)) ** 2)
    return torch.from_numpy(window.astype(np.float32))

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

            # nnet.module.reset_buffers()
            nnet.reset_buffers()
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

            length = inputs.shape[-1]
            start = 0
            win_len = 512
            hop_len = 256
            outputs = 0

            while (start + win_len <= length):
                inputs_frame = inputs[start:start+win_len]
                laec_echo_frame = laec_echo[start:start+win_len]
                start = start + hop_len
                outputs_frame = nnet(inputs_frame, laec_echo_frame)["specs"]
                if start == hop_len:
                    outputs = outputs_frame
                else:
                    outputs = torch.concat([outputs, outputs_frame], dim=-2)
                    # zero_pad = torch.zeros(outputs.shape[0], hop_len, device=outputs.device)
                    # outputs = torch.concat([outputs, zero_pad], dim=-1)
                    # outputs[:, -win_len:] += outputs_frame

            window = vorbis_window(512).to(outputs.device)
            outputs_stream = torch.istft(outputs.permute(0, 2, 1), n_fft=512, hop_length=256, win_length=512, window=window, return_complex=False, center=False)
            outputs_wav = outputs_stream.squeeze(0)

            out = outputs_wav.detach().cpu().numpy()

            if not os.path.exists(os.path.join(conf["save"]["dir"], mode)):
                os.makedirs(os.path.join(conf["save"]["dir"], mode))
            save_path = os.path.join(conf["save"]["dir"], mode, egs["utt_id"])
            sf.write(save_path, out, conf["save"]["sample_rate"])

def run(args):
    torch.set_printoptions(profile="full")
    # LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    # WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    # WORLD_RANK = int(os.environ['RANK'])    
    # dist.init_process_group(args.backend, rank=WORLD_RANK, world_size=WORLD_SIZE)    
    # torch.cuda.set_device(LOCAL_RANK)
    # device = torch.device('cuda', LOCAL_RANK)
    device = torch.device('cpu')
    # print(f"[{os.getpid()}] using device: {device}", torch.cuda.current_device(), "local rank", LOCAL_RANK)    
    
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
    # nnet = DistributedDataParallel(nnet, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True) 

    checkpoint_dir = Path(conf["test"]["checkpoint"])
    if os.path.isfile(checkpoint_dir):
        cpt_fname = checkpoint_dir
    elif os.path.isdir(checkpoint_dir):
        cpt_fname = checkpoint_dir / "best.pt.tar"
    # nnet.reload_spk()
    cpt = th.load(cpt_fname, map_location="cpu")

    state_dict = cpt['model_state_dict']
    # 去掉 'module.' 前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        # 只去掉 'module.' 前缀，但保留 'rnnoise_module.' 前缀中的 'module'
        if key.startswith('module.') and not key.startswith('rnnoise_module.'):
            new_key = key.replace('module.', '', 1)  # 只替换第一个 'module.'
        else:
            new_key = key  # 如果不符合条件则不修改
        new_state_dict[new_key] = value

    nnet.load_state_dict(new_state_dict, strict=False)
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