import librosa
import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch.utils.data as tud
from torch.utils.data.distributed import DistributedSampler
import json
import os
import sys
import re
import torch
import tqdm
import torch.distributed as dist

sys.path.append(os.path.dirname(sys.path[0]))
from libs.non_linear import echoclip
from libs.non_linear import nonlinear1, nonlinear2, nonlinear3, echoclip, reverb
from libs.utils import trunc_to_len, tail_to_len, delay
import time
import scipy.signal as sps
import random
import math
from linear_model.pfdkf_zsm import pfdkf
eps = np.finfo(np.float32).eps

"""AEC dataset
three channel needed, ref, mic, and near
ref is input reference signal.
mic is input mixture signal,
near is label signal.
"""

def add_reverb(cln_wav, rir_wav):
    # cln_wav: L
    # rir_wav: L
    wav_tgt = sps.oaconvolve(cln_wav, rir_wav)
    wav_tgt = wav_tgt[:cln_wav.shape[0]]
    return wav_tgt

def rms(data):
    """
    calc rms of wav
    """
    energy = data ** 2
    max_e = np.max(energy)
    low_thres = max_e * (10**(-50/10)) # to filter lower than 50dB 
    rms = np.mean(energy[energy >= low_thres])
    #rms = np.mean(energy)
    return rms

def snr_mix(clean, noise, snr):
    clean_rms = rms(clean)
    clean_rms = np.maximum(clean_rms, eps)
    noise_rms = rms(noise)
    noise_rms = np.maximum(noise_rms, eps)
    k = math.sqrt(clean_rms / (10**(snr/20) * noise_rms))
    new_noise = noise * k
    return new_noise

class AECDataset(object):
    def __init__(self,
                 config,
                 repeat=1,
                 seg_length=10,
                 echo_delay=False,
                 rnn_random_chance=50,
                 rnn_random_num=3,
            ):
        with open(config['speech_list'], 'r') as fid:
            self.nearend_lst = [line.strip() for line in fid.readlines()]
        self.nearend_lst *= repeat
        with open(config["noise_list"], 'r') as fid:
            self.noise_lst = [line.strip() for line in fid.readlines()]
        with open(config["rir_list"], 'r') as fid:
            self.rir_list = [line.strip() for line in fid.readlines()]
        self.seg_length = seg_length*16000
        self.echo_delay = echo_delay
        self.snr_low = config['snr_low']
        self.snr_high = config['snr_high']
        self.ser_low = config['ser_low']
        self.ser_high = config['ser_high']
        self.randstates = [np.random.RandomState(idx) for idx in range(3000)]

        self.rnn_random_num = rnn_random_num
        self.rnn_random_chance = rnn_random_chance

    def __len__(self):
        return len(self.nearend_lst)
    
    def load_wav(self, path, target_sr=16000):
        sig, sr = sf.read(path)
        # 若为多通道音频，只取第一个通道
        if len(sig.shape) > 1:
            sig = sig[:, 0]
        # 如果采样率不是16k，则重采样到16k
        if sr is not target_sr:
            sig = librosa.resample(sig, orig_sr=sr, target_sr=target_sr)
        return sig
    
    def add_distortion_and_reverb(self, ref_sig, rir, set_echo_clip):
        if np.random.uniform(0, 1) < 0.1:
            non_type = np.random.uniform(0, 1)
            if non_type < 0.3:
                echo_ref_sig = nonlinear1(ref_sig)
            elif non_type < 0.7:
                echo_ref_sig = nonlinear2(ref_sig)
            else:
                # add nonlinear by clip
                echo_ref_sig = nonlinear3(ref_sig)
            if set_echo_clip:
                echo_ref_sig, _ = reverb(echo_ref_sig, rir)
            else:
                _, echo_ref_sig = reverb(echo_ref_sig, rir)
        else:
            if set_echo_clip:
                echo_ref_sig, _ = reverb(ref_sig, rir)
            else:
                _, echo_ref_sig = reverb(ref_sig, rir)
        if set_echo_clip:
            echo_ref_sig = echoclip(echo_ref_sig)
        return echo_ref_sig
    
    def set_echo_delay(self, echo_ref_sig, set_echo_delay=True):
        if set_echo_delay:
            do_delay = np.random.random()
            if do_delay < 0.5:
                echo_ref_sig = delay(echo_ref_sig)
        return echo_ref_sig
    
    def choice(self, data_list: list):
        data_list_len = len(data_list)
        choice_idx = random.randint(0, data_list_len - 1)
        pth = data_list[choice_idx]
        return pth

    def __getitem__(self, index):
        data_list_len = len(self.nearend_lst)
        randstate = self.randstates[(index + 11) % 3000]
        # 近端语音
        near_sig_pth = self.nearend_lst[index]
        near_spk_id = os.path.basename(near_sig_pth)
        label_sig = self.load_wav(near_sig_pth, 16000)
        label_sig = trunc_to_len(label_sig, self.seg_length, 0.3)
        # 远端语音
        ref_sig_pth = self.choice(self.nearend_lst)
        ref_spk_id = os.path.basename(ref_sig_pth)
        while near_spk_id == ref_spk_id:
            ref_sig_pth = self.choice(self.nearend_lst)
            ref_spk_id = os.path.basename(ref_sig_pth)
        ref_sig = self.load_wav(ref_sig_pth, 16000)
        ref_sig = trunc_to_len(ref_sig, self.seg_length, -1)
        # 噪声
        noise_sig_pth = self.choice(self.noise_lst)
        # 远端语音的RIR
        rir_sig_pth = self.choice(self.rir_list)
        rir_id = os.path.basename(rir_sig_pth)
        rir = self.load_wav(rir_sig_pth, 16000)
        # 近端语音的RIR
        rir_sig_pth2 = self.choice(self.rir_list)
        rir2_id = os.path.basename(rir_sig_pth2)
        while rir_id == rir2_id:
            rir_sig_pth2 = self.choice(self.rir_list)
            rir2_id = os.path.basename(rir_sig_pth2)
        rir2 = self.load_wav(rir_sig_pth2, 16000)

        scale1 = np.random.uniform(0.15, 0.95)
        scale2 = np.random.uniform(0.15, 0.95)
        ser = np.random.uniform(self.ser_low, self.ser_high)
        snr = np.random.uniform(self.snr_low, self.snr_high)
        
        # 仿造不同场景情况下的数据
        set_noise_zeros = True
        set_ref_zeros = False
        set_near_rir = np.random.random() < 0.5
        set_echo_db_sub = np.random.random() < 0.3 if not set_ref_zeros else False
        set_near_zeros = np.random.random() < 0.1 if not set_ref_zeros else False
        set_echo_clip = np.random.random() < 0.2 if not set_ref_zeros else False
        # usually in ser < 0...
        set_echo_clip = set_echo_clip and ser < 0
        if set_echo_clip:
            scale1 = np.random.uniform(0.6, 1)
            scale2 = np.random.uniform(0.15, 0.5)        
        
        if set_near_rir:
            label_sig = add_reverb(label_sig, rir2)
        # add nonlinear for echo signal before reverb
        echo_ref_sig = self.add_distortion_and_reverb(ref_sig, rir, set_echo_clip)
        # delay echo
        echo_ref_sig = self.set_echo_delay(echo_ref_sig)
        noise_sig, sr = sf.read(noise_sig_pth)
        while sr != 16000:
            if sr == 48000:
                noise_sig = librosa.resample(noise_sig, orig_sr=sr, target_sr=16000)
                sr = 16000
            else:
                noise_sig_pth = np.random.choice(self.noise_lst)
                noise_sig, sr = sf.read(noise_sig_pth)
        noise_sig = trunc_to_len(noise_sig, self.seg_length)
        # dynamic ser snr generator.
        echo_ref_sig = snr_mix(label_sig, echo_ref_sig, ser)
        noise_sig = snr_mix(label_sig, noise_sig, snr)
        
        # noise stat
        noise_stat = (np.linalg.norm(noise_sig*32768, 2)**2) / len(noise_sig)
        if noise_stat < 1e-2:
            set_noise_zeros = True
        # ref stat
        ref_stat = (np.linalg.norm(ref_sig*32768, 2)**2) / len(ref_sig)
        if ref_stat < 1e-2:
            set_ref_zeros = True
        # echo stat
        echo_stat = (np.linalg.norm(echo_ref_sig*32768, 2)**2) / \
            len(echo_ref_sig)
        if echo_stat < 1e-2:
            set_ref_zeros = True
        
        if set_noise_zeros:
            noise_sig = np.zeros_like(noise_sig)
        if set_ref_zeros:
            ref_sig = np.zeros_like(ref_sig)
            echo_ref_sig = np.zeros_like(echo_ref_sig)
        if set_near_zeros:
            label_sig = np.zeros_like(label_sig)
        if set_echo_db_sub and not set_ref_zeros:
            mod = np.random.random() < 0.5
            sub_db = np.random.uniform(15, 30)
            if mod:
                end = int(np.random.uniform(2, 4)*sr)
                echo_ref_sig[:end] = echo_ref_sig[:end]*10**(-sub_db/20)
            else:
                start = int(np.random.uniform(1, 4)*sr)
                end = int(np.random.uniform(5, 7)*sr)
                echo_ref_sig[start:end] = echo_ref_sig[start:end]*10**(-sub_db/20)
        
        simu_mic_sig = echo_ref_sig + label_sig + noise_sig

        den = np.linalg.norm(simu_mic_sig, np.inf)
        if den != 0:
            coef = scale1 / den
            simu_mic_sig *= coef
            label_sig *= coef
            echo_ref_sig *= coef

        if ref_stat > 1e-2:
            den = np.linalg.norm(ref_sig, np.inf)
            if den != 0:
                coef = scale2 / den
                ref_sig *= coef
        
        laec_outputs, laec_echo = pfdkf(ref_sig, simu_mic_sig)

        return {
            "mix": simu_mic_sig.astype(np.float32),
            "laec_out": laec_outputs.astype(np.float32),
            "laec_echo": laec_echo.astype(np.float32),
            "far": ref_sig.astype(np.float32), 
            "echo_ref_sig": echo_ref_sig.astype(np.float32),
            "label": label_sig.astype(np.float32),
        }

def make_loader(config_dir, batch_size=1, shuffle=True,
                num_workers=0, repeat=1,
                seg_len=10, echo_delay=True,
                ):
    print(f'config_dir: {config_dir}')
    with open(config_dir, 'r') as fid:
        config = json.load(fid)
    dataset = AECDataset(config, repeat=repeat,
                         seg_length=seg_len, echo_delay=echo_delay,
                         )
    sampler = DistributedSampler(dataset)
    loader = tud.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True,
        # shuffle=shuffle,
        pin_memory=True,
    )
    return loader

def make_loader_test(config_dir, batch_size=1, shuffle=True,
                num_workers=0, repeat=1,
                seg_len=10, echo_delay=True,
                ):
    with open(config_dir, 'r') as fid:
        config = json.load(fid)
    dataset = AECDataset(config, repeat=repeat,
                         seg_length=seg_len, echo_delay=echo_delay,
                         )
    loader = tud.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=shuffle,
        pin_memory=True,
    )
    print(f'total samples is {len(loader)}')
    return loader

def test_loader():
    torch.set_printoptions(profile="full")

    config_dir = "/home/node25_tmpdata/xcli/percepnet/train/big_data/config_tr.json"

    batch_size = 1
    sample_rate = 16000

    loader = make_loader_test(config_dir, batch_size=batch_size, repeat=1, num_workers=16, seg_len=10)

    cnt = 0
    for idx, egs in enumerate(loader):
        cnt = cnt + 1
        print('cnt: {}'.format(cnt))
        if cnt == 10000:
            break
        print('egs["mix"].shape: ', egs["mix"].shape)
        print('egs["far"].shape: ', egs["far"].shape)

    print('done!')


if __name__ == "__main__":
    test_loader()