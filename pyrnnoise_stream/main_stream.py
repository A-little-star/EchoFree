"""
Author: xcli
Email: 2621939373@qq.com
Date: 2024/8/11
Description: wav->BFCC转换模块，使用pytorch框架实现
Version: 1.0
"""
import torch
import torch.nn.functional as F
import numpy as np

def rnnoise_in(
        domain, 
        inputs,
        inputs_buffer,
        buffer_idx,
        Fbank, 
        n_fft=None, 
        hop_length=None, 
        win_length=None, 
        window=None,
        up_scale=1.0
        ):

    if (isinstance(inputs, torch.Tensor)):
        inputs = inputs.to(torch.float32) * up_scale
    elif (isinstance(inputs, np.ndarray)):
        inputs = torch.from_numpy(inputs).to(torch.float32) * up_scale
    else:
        print('Error type wav! (in rnnoise_in of main)')
    
    if domain == "Time":
        X = torch.stft(inputs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True, center=False)
        X = X.permute(0, 2, 1)  # [B, T, F]
    elif domain == "Frequency":
        X = inputs
    else:
        print(f"Undefined domain! (in rnnoise_in of main)")

    mag_square_X = torch.abs(X) ** 2
    Ex = torch.matmul(mag_square_X, Fbank.T)

    out_features = torch.log10(1e-2 + Ex)

    out_features[:, :, 0] -= 12
    out_features[:, :, 1] -= 4

    if inputs_buffer[buffer_idx] is None:
        inputs_buffer[buffer_idx] = torch.zeros(out_features.shape[0], 2, out_features.shape[2], device=out_features.device, dtype=out_features.dtype)
    padded_out_features = torch.concat([inputs_buffer[buffer_idx], out_features[:, :, :]], dim=1)
    ceps_1 = padded_out_features[:, 2:, :] - padded_out_features[:, :-2, :]
    ceps_2 = padded_out_features[:, 2:, :] - 2*padded_out_features[:, 1:-1, :] + padded_out_features[:, :-2, :]
    out_features = torch.cat([out_features, ceps_1, ceps_2], dim=-1)
    
    inputs_buffer[buffer_idx] = padded_out_features[:, -2:, :]

    return out_features

def rnnoise_getgains(
        domain, 
        inputs, 
        labels, 
        Fbank, 
        n_fft=None, 
        hop_length=None, 
        win_length=None, 
        window=None,
        up_scale=1.0
        ):

    if (isinstance(inputs, torch.Tensor)):
        inputs = inputs.to(torch.float32) * up_scale
        labels = labels.to(torch.float32) * up_scale
    elif (isinstance(inputs, np.ndarray)):
        inputs = torch.from_numpy(inputs).to(torch.float32) * up_scale
        labels = torch.from_numpy(labels).to(torch.float32) * up_scale
    else:
        print('Error type wav! (in rnnoise_getgains of main)')

    if domain == "Time":
        X = torch.stft(inputs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True, center=False)
        X = X.permute(0, 2, 1)  # [B, T, F]
        Y = torch.stft(labels, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True, center=False)
        Y = Y.permute(0, 2, 1)
    elif domain == "Frequency":
        X = inputs
        Y = labels
    else:
        print(f"Undefined domain! (in rnnoise_getgains of main)")

    B, T, F_ = X.shape
    noisyEng = torch.abs(X) ** 2
    noisyEng = noisyEng[:, :, :F_ // 2]
    xyCor = X.real * Y.real + X.imag * Y.imag
    xyCor = xyCor[:, :, :F_ // 2]
    gg = xyCor / (noisyEng + 1e-3)
    gg[gg > 1] = 1
    gg[gg < -1] = -1
    Y[:, :, :F_ // 2] = gg * Y[:, :, :F_ // 2]

    mag_square_X = torch.abs(X) ** 2
    mag_square_Y = torch.abs(Y) ** 2
    Ex = torch.matmul(mag_square_X, Fbank.T)
    Ey = torch.matmul(mag_square_Y, Fbank.T)

    out_gains = torch.sqrt(Ey / (Ex + 1e-3))
    out_gains[out_gains > 1] = 1
    out_gains[(Ex < 5e-2) & (Ey < 5e-2)] = -1

    return out_gains

def rnnoise_out(
        domain, 
        inputs,
        gains, 
        Fbank, 
        n_fft=None, 
        hop_length=None, 
        win_length=None, 
        window=None,
        up_scale=1.0
        ):

    if (isinstance(inputs, torch.Tensor)):
        inputs = inputs.to(torch.float32) * up_scale
    elif (isinstance(inputs, np.ndarray)):
        inputs = torch.from_numpy(inputs).to(torch.float32) * up_scale
    else:
        print('Error type wav! (in rnnoise_out of main)')
    
    # buffer_size = win_length - hop_length
    # if inputs_buffer is None:
    #     inputs_buffer = torch.zeros(inputs.shape[0], buffer_size, device=inputs.device, dtype=inputs.dtype)
    # inputs = torch.concat([inputs_buffer, inputs], dim=1)
    # inputs_buffer = inputs[:, -buffer_size:]

    if domain == "Time":
        X = torch.stft(inputs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True, center=False)
        X = X.permute(0, 2, 1)  # [B, T, F]
    elif domain == "Frequency":
        X = inputs
    else:
        print(f"Undefined domain! (in rnnoise_out of main)")
    
    mask = torch.matmul(gains, Fbank)

    out_spec = X * mask
    
    out_spec /= up_scale
    out_wav = torch.istft(out_spec.permute(0, 2, 1), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=False, center=False)

    return out_spec, out_wav