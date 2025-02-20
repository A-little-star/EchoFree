import sys
import torchaudio
import torch
import numpy as np
sys.path.append("/home/work_nfs7/xcli/work/percepnet/c_spec/libs/fd_240_120_256_py38")
from percepnet import percepnet_train, percepnet_postProcess
import soundfile as sf
import os
import librosa

def vorbis_window(n):
    """Generate a Vorbis window of length n."""
    window = np.zeros(n)
    for i in range(n):
        window[i] = np.sin((np.pi / 2) * (np.sin(np.pi * (i + 0.5) / n)) ** 2)
    return window

def rnnoise_in(inputs, labels, nb_features=64, nb_bands=50, 
                  n_fft=256, hop_length=120, win_length=240):

    if (isinstance(inputs, torch.Tensor)):
        inputs = inputs.numpy().astype('float32')
        labels = labels.numpy().astype('float32')
    elif (isinstance(inputs, np.ndarray)):
        inputs = inputs.astype('float32')
        labels = labels.astype('float32')
    else:
        print('Error type wav! (in wav2rnnoisein)')
    
    window = vorbis_window(win_length)
    
    inputs_spec = librosa.stft(inputs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window).squeeze()
    inputs_spec = np.transpose(inputs_spec)
    inputs_real = np.real(inputs_spec)
    inputs_imag = np.imag(inputs_spec)

    labels_spec = librosa.stft(labels, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window).squeeze()
    labels_spec = np.transpose(labels_spec)
    labels_real = np.real(labels_spec)
    labels_imag = np.imag(labels_spec)

    rnnoise_inp = np.concatenate((labels_real.flatten(), labels_imag.flatten(), inputs_real.flatten(), inputs_imag.flatten()))

    features = percepnet_train(rnnoise_inp).reshape([-1, nb_features+nb_bands]).astype('float32')
    features[:, nb_features:nb_features+nb_bands] = np.clip(features[:, nb_features:nb_features+nb_bands], 0, 1)
    inputs_features = features[:, :nb_features]
    gains = features[:, nb_features:nb_features+nb_bands]
    vad = 1
    return inputs_features, gains, vad

def rnnoise_out(inputs, gains,
                n_fft=256, hop_length=120, win_length=240):
    if (isinstance(inputs, torch.Tensor)):
        inputs = inputs.numpy().astype('float32')
    elif (isinstance(inputs, np.ndarray)):
        inputs = inputs.astype('float32')
    else:
        print('Error type wav! (in rnnoise_out)')

    window = vorbis_window(win_length)
    inputs_spec = librosa.stft(inputs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window).squeeze()
    inputs_spec = np.transpose(inputs_spec)
    inputs_real = np.real(inputs_spec)
    inputs_imag = np.imag(inputs_spec)

    outputs = percepnet_postProcess(np.concatenate((gains, inputs_real, inputs_imag), -1))
    
    freq_size = n_fft // 2 + 1
    window = vorbis_window(win_length)
    outputs = np.reshape(outputs, (-1, freq_size))
    outputs_real = outputs[0::2, :]
    outputs_imag = outputs[1::2, :]
    outputs_spec = outputs_real + 1j * outputs_imag

    outputs_wav = librosa.istft(np.transpose(outputs_spec), n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window).astype('float32')
    return outputs_wav