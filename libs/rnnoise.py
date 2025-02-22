import sys
import torchaudio
import torch
import numpy as np
sys.path.append("/home/work_nfs7/xcli/work/percepnet/c_kiss_fft/libs/fd_240_120_256_py38_34")
from percepnet import percepnet_train, percepnet_featExtract, percepnet_postProcess
import soundfile as sf
import os
import librosa

def rnnoise_in(inputs, labels, nb_features=64, nb_bands=50):
    if (isinstance(inputs, torch.Tensor)):
        inputs = inputs.numpy().astype('float32')*32768
        labels = labels.numpy().astype('float32')*32768
    elif (isinstance(inputs, np.ndarray)):
        inputs = inputs.astype('float32')*32768
        labels = labels.astype('float32')*32768
    else:
        print('Error type wav! (in rnnoise_in)')

    inp = np.concatenate((labels, inputs), axis=-1)
    inp = np.squeeze(inp)

    features = percepnet_train(inp).reshape([-1, nb_features + nb_bands]).astype('float32')
    inputs_features = features[:, :nb_features]
    gains = features[:, nb_features:nb_features+nb_bands]
    # vad = features[:, -1]
    vad = 0
    return inputs_features, gains, vad

def rnnoise_out(inputs, gains):
    if (isinstance(inputs, torch.Tensor)):
        inputs = inputs.numpy().astype('float32')*32768
    elif (isinstance(inputs, np.ndarray)):
        inputs = inputs.astype('float32')*32768
    else:
        print('Error type wav! (in rnnoise_out)')

    featExtract_output = percepnet_featExtract(inputs.squeeze()).reshape([-1, 129*2]).astype('float32')

    postProcess_output = percepnet_postProcess(np.concatenate((gains, featExtract_output),1)).astype('float32')/32768
    return postProcess_output