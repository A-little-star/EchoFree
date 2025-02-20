import librosa
import torch as th
import numpy as np
import soundfile as sf

import sys, os
sys.path.append(os.path.dirname(__file__))
# from speex_linear.lp_or_tde import LP_or_TDE
from linear_model.pfdkf_zsm import pfdkf

def audio(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        wave_data = librosa.resample(wave_data, orig_sr=sr, target_sr=fs)
    return wave_data

def get_firstchannel_read(path, fs=16000):
    wave_data, sr = sf.read(path)
    if sr != fs:
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
        wave_data = librosa.resample(wave_data, orig_sr=sr, target_sr=fs)
        if len(wave_data.shape) != 1:
            wave_data = wave_data.transpose((1, 0))
    if len(wave_data.shape) > 1:
        wave_data = wave_data[:, 0]
    return wave_data

def parse_scp(scp, path_list):
    with open(scp) as fid: 
        for line in fid:
            tmp = line.strip().split()
            if len(tmp) > 1:
                path_list.append({"inputs": tmp[0], "duration": tmp[1]})
            else:
                path_list.append({"inputs": tmp[0]})

class DataReader(object):
    def __init__(self, filename, sample_rate): #, aux_segment): # filename是不带id的待解码音频，noisy_id是带id的带解码音频，clean是带id的注册音频
        self.file_list = []
        parse_scp(filename, self.file_list)
        self.sample_rate = sample_rate

        # self.aux_segment_length = aux_segment * sample_rate

    def extract_feature(self, path):
        mic_path = path["inputs"]
        utt_id = mic_path.split("/")[-1]
        ref_path = mic_path.replace("mic.wav", "lpb.wav")

        mic = get_firstchannel_read(mic_path, self.sample_rate).astype(np.float32)
        ref = get_firstchannel_read(ref_path, self.sample_rate).astype(np.float32)

        max_mix_norm = np.max(np.abs(mic))
        if max_mix_norm == 0:
            max_mix_norm = 1
        
        min_len = min(mic.shape[0], ref.shape[0])
        mic = mic[:min_len]
        ref = ref[:min_len]
        
        # mic, echo, align_ref = LP_or_TDE(mic, ref, mode='lp')
        laec_outputs, laec_echo = pfdkf(ref, mic)

        min_len = min(laec_outputs.shape[0], ref.shape[0])
        mic = mic[:min_len]
        laec_outputs = laec_outputs[:min_len]
        laec_echo = laec_echo[:min_len]
        ref = ref[:min_len]

        inputs_mic = np.reshape(mic, [1, mic.shape[0]])
        inputs_laec_outputs = np.reshape(laec_outputs, [1, laec_outputs.shape[0]]).astype(np.float32)
        inputs_laec_echo = np.reshape(laec_echo, [1, laec_echo.shape[0]]).astype(np.float32)
        inputs_ref = np.reshape(ref, [1, ref.shape[0]]).astype(np.float32)
        
        inputs_mic = th.from_numpy(inputs_mic)
        inputs_laec_outputs = th.from_numpy(inputs_laec_outputs)
        inputs_laec_echo = th.from_numpy(inputs_laec_echo)
        inputs_ref = th.from_numpy(inputs_ref)
        
        egs = {
            "mix": inputs_mic,
            "ref": inputs_ref,
            "laec_out": inputs_laec_outputs,
            "laec_echo": inputs_laec_echo,
            "utt_id": utt_id,
            "max_norm": max_mix_norm
        }
        return egs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])
