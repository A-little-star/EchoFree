import sys, os
sys.path.append(os.path.dirname(__file__))
from vad import SingleChannelVAD
import numpy as np


def ideal_dtd(label):
    vad_label = SingleChannelVAD(zeroing=False)
    vad_label.process(label)
    vad_stat = vad_label.get_vad_stat()
    # frames = (len(label) - 320)//160 + 1
    # vad_samps = np.zeros_like(label)
    # for i in range(frames):
    #     vad_samps[i*160:i*160+320] = int(vad_stat[i])
    # return vad_samps/2
    return np.int32(vad_stat)


if __name__ == "__main__":
    # import soundfile as sf
    # mic, sr = sf.read(
    #     "/mnt/shimin.zshm/datasets/AEC-Challenge/datasets/synthetic/nearend_mic_signal/nearend_mic_fileid_100.wav")
    # NLFE, sr = sf.read(
    #     "/mnt/shimin.zshm/datasets/AEC-Challenge/datasets/synthetic/echo_signal/echo_fileid_100.wav")
    
    # label, sr = sf.read(
    #     "/mnt/shimin.zshm/datasets/AEC-Challenge/datasets/synthetic/nearend_speech/nearend_speech_fileid_100.wav")
    # should be different from reference and estimate signal.
    label = np.zeros((1,16000))
    result = ideal_dtd(label)
    print(result, result.shape)
    # sums = np.stack([label, result], axis=1)

    # sf.write('p1.wav', sums, 16000)
