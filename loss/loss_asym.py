import torch
import torch.nn as nn
import numpy as np
from utils.torch_stft import pad_audio_torch, unpad_audio_torch

eps = np.finfo(np.float32).eps

    
class HybridLoss(nn.Module):
    def __init__(self, stft):
        super().__init__()
        self.stft = stft
        self.nfft = stft.nfft
        self.hop = stft.hop
        self.win = stft.win

    def forward(self, pred_stft, true_stft, pred_pad_len, true_pad_len):
        device = pred_stft.device

        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag**(0.7))
        pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
        true_real_c = true_stft_real / (true_mag**(0.7))
        true_imag_c = true_stft_imag / (true_mag**(0.7))
        real_loss = nn.MSELoss()(pred_real_c, true_real_c)
        imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
        mag_loss = nn.MSELoss()(pred_mag**(0.3), true_mag**(0.3))
        
        y_pred = torch.istft(pred_stft_real+1j*pred_stft_imag, self.nfft, self.hop, self.win, window=torch.hann_window(self.win).pow(0.5).to(device))
        y_true = torch.istft(true_stft_real+1j*true_stft_imag, self.nfft, self.hop, self.win, window=torch.hann_window(self.win).pow(0.5).to(device))

        #! unpad
        y_pred = unpad_audio_torch(y_pred, pred_pad_len)
        y_true = unpad_audio_torch(y_true, true_pad_len)

        y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true), dim=-1, keepdim=True) + 1e-8)

        sisnr =  - torch.log10(torch.norm(y_true, dim=-1, keepdim=True)**2 / (torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2 + 1e-8) + 1e-8).mean()

        asym_loss = self.get_asym_loss(pred_mag, true_mag)

        return 30*(real_loss + imag_loss) + 70*mag_loss + sisnr + asym_loss
    
    def get_asym_loss(self, est_mag, ref_mag, lamda=0.5, eps=1e-8, use_first=False, fft_cfg=None):

        press_est_mag = est_mag**lamda
        press_ref_mag = ref_mag**lamda

        asym_mag_loss = torch.square(torch.clamp(press_est_mag - press_ref_mag, max=0))

        loss = asym_mag_loss
        
        N, F, T = loss.shape
        loss = torch.mean(loss) * F
        return loss


if __name__ == "__main__":
    loss_func = HybridLoss(1)

    pred_stft = torch.randn(1, 257, 63, 2)
    true_stft = torch.randn(1, 257, 63, 2)
    loss = loss_func(pred_stft, true_stft)
    print(loss)
