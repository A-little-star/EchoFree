import numpy as np
import torch

# 定义常量
WIN_LEN = 240
HOP_LEN = 120
FFT_LEN = 256
FRAME_SIZE_SHIFT = 0  # 根据需要调整
FREQ_SIZE = FFT_LEN // 2 + 1

NB_BANDS = 50  # 假设这个值与eband5ms数组长度一致
NB_DELTA_CEPS = 6
NB_FEATURES = NB_BANDS + 2*NB_DELTA_CEPS + 2

CEPS_MEM = 8

eband5ms = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
  20, 21, 22, 23, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 45, 48, 52,
  54, 57, 61, 65, 69, 73, 77, 83, 89, 95, 101, 109, 125
]

def SQUARE(x):
    return x * x

class DenoiseState():
  def __init__(self, batch_size=192):
    self.analysis_mem = torch.zeros(WIN_LEN - HOP_LEN, dtype=torch.float32)
    self.cepstral_mem = torch.zeros((CEPS_MEM, batch_size, NB_BANDS), dtype=torch.float32)
    self.memid = torch.tensor(0)