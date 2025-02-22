"""
Author: xcli
Email: 2621939373@qq.com
Date: 2024/8/11
Description: wav->BFCC转换模块，使用pytorch框架实现
Version: 1.0
References: https://github.com/LXP-Never/perception_scale/blob/main/RNNoise_band_energy.py：尺度变换与滤波器
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import sys,os
sys.path.append(os.path.dirname(__file__))

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(project_root)

from pyrnnoise_stream.main_stream import rnnoise_in, rnnoise_out, rnnoise_getgains

class MelFilterBank():
    def __init__(self, nfilter=32, nfft=512, sr=16000,
                 lowfreq=0, highfreq=None, transpose=False):
        """
        :param nfilter: filterbank中的滤波器数量
        :param nfft: FFT size
        :param sr: 采样率
        :param lowfreq: Mel-filter的最低频带边缘
        :param highfreq: Mel-filter的最高频带边缘，默认samplerate/2
        """
        self.nfilter = nfilter
        self.freq_bins = int(nfft / 2 + 1)
        highfreq = highfreq or sr / 2
        self.transpose = transpose

        # 按梅尔均匀间隔计算 点
        lowmel, highmel = self.hz2mel(lowfreq), self.hz2mel(highfreq)
        melpoints = np.linspace(lowmel, highmel, nfilter)
        hz_points = self.mel2hz(melpoints)  # 将mel频率再转到hz频率

        self.bins_list = np.floor(hz_points * nfft / sr).astype(np.int32)
        # 这里先将子带硬编码下来，后面可以尝试不同的子带划分
        # self.bins_list = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        #                             20, 21, 22, 23, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 45, 48, 52,
        #                             54, 57, 61, 65, 69, 73, 77, 83, 89, 95, 101, 109, 125])
        self.bins_list = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48,  50,  52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,
                                    74,  76,  78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,
                                    102, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 146, 152,
                                    158, 164, 170, 176, 182, 188, 194, 200, 208, 216, 224, 232, 240, 256])
        # [  0   1   3   5   8  10  13  15  18  22  25  29  33  38  42  48  53  59
        #   66  73  80  88  97 107 117 128 140 153 167 182 198 216 235 256]

    def hz2mel(self, hz, approach="Oshaghnessy"):
        """ Hz to Mels """
        return {
            "Oshaghnessy": 2595 * np.log10(1 + hz / 700.0),
            "Lindsay": 2410 * np.log10(1 + hz / 625),
        }[approach]

    def mel2hz(self, mel, approach="Oshaghnessy"):
        """ Mels to HZ """
        return {
            "Oshaghnessy": 700 * (10 ** (mel / 2595.0) - 1),
            "Lindsay": 625 * (10 ** (mel / 2410) - 1),
        }[approach]

    def get_filter_bank(self):
        # RNNoise的三角滤波器，打叉画法
        fbank = np.zeros((self.nfilter, self.freq_bins))  # (M,F)
        for i in range(self.nfilter - 1):
            band_size = (self.bins_list[i + 1] - self.bins_list[i])
            for j in range(band_size):
                frac = j / band_size
                fbank[i, self.bins_list[i] + j] = 1 - frac  # 降
                fbank[i + 1, self.bins_list[i] + j] = frac  # 升
        # 第一个band和最后一个band的窗只有一半因而能量乘以2
        fbank[0] *= 2
        fbank[-1] *= 2
        return torch.from_numpy(fbank.astype(np.float32))

    def interp_band_gain(self, gain):
        # gain (M,T)
        gain_interp = np.zeros((self.freq_bins, gain.shape[-1]))  # (F,T)
        for i in range(self.nfilter - 1):
            band_size = (self.bins_list[i + 1] - self.bins_list[i])
            for j in range(band_size):
                frac = j / band_size
                gain_interp[self.bins_list[i] + j] = (1 - frac) * gain[i] + \
                                                     frac * gain[i + 1]

        return gain_interp

def vorbis_window(n):
    """Generate a Vorbis window of length n."""
    window = np.zeros(n)
    for i in range(n):
        window[i] = np.sin((np.pi / 2) * (np.sin(np.pi * (i + 0.5) / n)) ** 2)
    return torch.from_numpy(window.astype(np.float32))

class RnnoiseModule(nn.Module):
    def __init__(self, n_fft, hop_len, win_len, up_scale, nfilter=50):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        fbank_class = MelFilterBank(nfilter=nfilter, nfft=n_fft,
                                sr=16000, lowfreq=0, highfreq=None, transpose=False)
        fbank = fbank_class.get_filter_bank()
        window = vorbis_window(win_len)
        self.buffer = [None, None]
        self.register_buffer('fbank', fbank)
        self.register_buffer('window', window)
        # self.register_buffer('buffer', buffer)
        self.up_scale=up_scale
    
    def reset_buffer(self):
        self.buffer = [None, None]
    
    def forward_transform(self, inputs, buffer_idx):
        '''
        该方法用于将输入的wav信号转换为BFCC特征（还包括一些额外特征）
        inputs: 混合语音的时域信号，以wav的形式输入 [B, L]
        '''
        out_features = rnnoise_in(
            domain="Time", 
            inputs=inputs,
            inputs_buffer=self.buffer,
            buffer_idx=buffer_idx,
            Fbank=self.fbank, 
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            up_scale=self.up_scale
        )

        out_features = out_features.unsqueeze(1)
        
        # shape: [B, C, T, D]
        return out_features

    
    def inverse_transform(self, inputs, gains):
        '''
        该方法用于将混合语音与增益进行处理，得到干净语音
        inputs: 混合语音的时域信号，以wav的形式输入 [B, L]
        gains:  增益，用于获取干净语音 [B, T, NB_BANDS]
        '''
        out_specs, out_wavs = rnnoise_out(
            domain="Time",
            inputs=inputs,
            gains=gains,
            Fbank=self.fbank,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            up_scale=self.up_scale
        )

        return out_specs, out_wavs
    
    def get_gains(self, inputs, labels):
        '''
        该方法用于计算理想的增益，需要输入混合语音和干净语音的wav信号
        inputs: 混合语音的时域信号，以wav的形式输入 [B, L]
        labels: 干净语音的时域信号，以wav的形式输入 [B, L]
        '''
        out_gains = rnnoise_getgains(
            domain="Time",
            inputs=inputs,
            labels=labels,
            Fbank=self.fbank,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window,
            up_scale=self.up_scale
        )

        return out_gains

if __name__=='__main__':
    model = RnnoiseModule(n_fft=512, hop_len=256, win_len=512, up_scale=64.0, nfilter=100)

    inputs = torch.randn(1, 512 + 256 * 1)
    labels = torch.randn(1, 512 + 256 * 1)

    inputs_features = model.forward_transform(inputs, 0)
    gains_ori = model.get_gains(inputs, labels)
    _, outputs_ori = model.inverse_transform(inputs, gains_ori)
    # print(f'gains_ori: {gains_ori}')
    print(f'wav_ori: {outputs_ori}')
    model.reset_buffer()

    
    outputs = 0
    start = 0
    win_len = 512
    hop_len = 256
    length = inputs.shape[-1]
    
    while (start + win_len <= length):
        inputs_frame = inputs[:, start:start+win_len]
        labels_frame = labels[:, start:start+win_len]
        start = start + hop_len
        inputs_features_frame = model.forward_transform(inputs_frame, 0)
        gains_frame = model.get_gains(inputs_frame, labels_frame)
        _, outputs_frame = model.inverse_transform(inputs_frame, gains_frame)
        print(_.shape)
        if start == hop_len:
            gains = gains_frame
            outputs = outputs_frame
        else:
            # outputs = torch.concat([outputs, outputs_frame], dim=-2)
            gains = torch.concat([gains, gains_frame], dim=-2)
            zero_pad = torch.zeros(outputs.shape[0], hop_len, device=outputs.device)
            outputs = torch.concat([outputs, zero_pad], dim=-1)
            outputs[:, -hop_len:] += outputs_frame[:, -hop_len]
    
    # print(f'gains: {gains}')
    # print(f'gains error: {(gains_ori - gains) / gains_ori}')
    print(f'max gains error: {torch.max((gains_ori - gains) / gains_ori)}')
    print(f'wav_stream: {outputs}')
    print(f'wavs error: {(outputs_ori - outputs) / outputs_ori}')
    print(f'max wavs error: {torch.max((outputs_ori - outputs) / outputs_ori)}')
    # print((outputs_ori - outputs) / outputs_ori)
    # print(torch.max((outputs_ori - outputs) / outputs_ori))