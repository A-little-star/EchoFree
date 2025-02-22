import sys,os
sys.path.append(os.path.dirname(__file__))
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import torch 
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from functools import partial

from pyrnnoise.rnnoise_module import RnnoiseModule

class DepthwiseConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: int,
                 padding: tuple,
                ):
        super().__init__()
        
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1)
        
    
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self,
                hidden_channels: int,
                kernel_size: tuple = (4, 3),
                stride: int = 1,
                casual: bool = True):
        super().__init__()
        self.causal = casual
        if self.causal:
            p_time = (kernel_size[0] - 1) 
        else:
            p_time = (kernel_size[0] - 1) // 2
        self.p_time = p_time
        p_freq = (kernel_size[1] - 1) // 2
        self.conv2d = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, (p_time, p_freq))
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ELU()
    
    def forward(self, x):
        # x in shape of [Batch, Channel, Time, Frequency] 
        z = self.conv2d(x)
        if self.causal:
            z = z[:,:,:-self.p_time,:]
        z = self.bn(z)
        z = self.act(z)
        y = z + x 
        return y

class SkipBlock(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride)
    
    def forward(self, encoder_feats, decoder_feats):
        proj_feats = self.conv(encoder_feats)
        proj_feats = encoder_feats
        f_en, f_de = proj_feats.shape[-1], decoder_feats.shape[-1]
        if f_en > f_de:
            decoder_feats = F.pad(decoder_feats, [0, f_en-f_de])    
        out = proj_feats + decoder_feats
        return out

class SubPixelConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 upscale_factor: int = 2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * upscale_factor, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)  # (N, out_channels * upscale_factor, t, f)

        # Reshape for pixel shuffle on frequency axis
        N, C, T, F = x.shape
        x = x.view(N, C // self.upscale_factor, self.upscale_factor, T, F)  # (N, out_channels, upscale_factor, t, f)
        x = x.permute(0, 1, 3, 4, 2)  # (N, out_channels, t, f, upscale_factor)
        x = x.contiguous().view(N, C // self.upscale_factor, T, F * self.upscale_factor)  # (N, out_channels, t, f * upscale_factor)

        return x

class EncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple = (4, 3),
                 stride: tuple = (1, 2),
                 causal: bool = True,
                 use_res: bool = True
                 ):
        super().__init__()
        self.causal = causal
        if self.causal:
            p_time = (kernel_size[0] - 1)  # (K-1)*D, in this case Dilaion=1
        else:
            p_time = (kernel_size[0] - 1) // 2
        self.p_time = p_time
        self.conv2d = DepthwiseConv2d(in_channels, out_channels, kernel_size, stride, (p_time, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()
        self.use_res = use_res
        if self.use_res:
            self.resblock = ResidualBlock(out_channels, casual=causal)
        
    def forward(self, x):
        z = self.conv2d(x)
        if self.causal:
            z = z[:,:,:-self.p_time,:]
        z = self.bn(z)
        z = self.act(z)
        if self.use_res:
            z = self.resblock(z)
        return z 

class BottleNeck(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional
        if bidirectional: 
            self.rnn = nn.GRU(hidden_dim, hidden_dim//2, batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    
    def forward(self, x):
        # x in shape of [Batch, Channel, Time, Frequency]
        x = x.permute(0, 2, 1, 3)
        B, T, C, _ = x.shape 
        x = x.reshape(B, T, -1).contiguous()
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.reshape(B, T, C, -1).permute(0, 2, 1, 3)
        return x 

class DecoderBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 upscale_factor: int = 2,
                 is_last: bool = False,
                 causal: bool = True,
                 use_res: bool = True):
        super().__init__()        
        self.skip = SkipBlock(in_channels)
        self.use_res = use_res
        if self.use_res:
            self.resblock = ResidualBlock(in_channels, casual=causal)        
        self.subpixconv = SubPixelConv(in_channels, out_channels, upscale_factor)
        self.is_last = is_last
        if not is_last:
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = nn.ELU()
    
    def forward(self, en, de):
        x = self.skip(en, de)
        if self.use_res:
            x = self.resblock(x)
        x = self.subpixconv(x)
        if not self.is_last:
            x = self.bn(x)
            x = self.act(x)
        return x 

class DeepVQES(nn.Module):
    '''Light-weight version without the align block
    '''
    def __init__(self,
                 in_dim: int = 64,
                 out_dim: int = 50,
                 casual: bool = True,
                 bidirectional: bool = False):
        super().__init__()
        self.casual = casual
        
        num_layers = 4
        
        encoder_mic_channels = [1, 8, 16, 24, 32]
        encoder_ref_channels = [1, 8]
        bottleneck_channels = 32 * 6
        decoder_channels = [32, 24, 16, 8, 1]

        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.num_layers = num_layers
        
        self.mic_encoders = nn.ModuleList()
        self.ref_encoders = nn.ModuleList()

        for idx in range(num_layers):
            if idx < 1:
                self.mic_encoders.append(
                    EncoderBlock(encoder_mic_channels[idx], encoder_mic_channels[idx+1], causal=casual, use_res=False)
                )
                self.ref_encoders.append(
                    EncoderBlock(encoder_ref_channels[idx], encoder_ref_channels[idx+1], causal=casual, use_res=False) 
                )
            elif idx == 1:
                self.mic_encoders.append(
                    EncoderBlock(encoder_mic_channels[idx]+encoder_ref_channels[idx], encoder_mic_channels[idx+1], causal=casual, use_res=False)
                )
            else:
                self.mic_encoders.append(
                    EncoderBlock(encoder_mic_channels[idx], encoder_mic_channels[idx+1], causal=casual, use_res=False)
                )
                                    
        self.bottleneck = BottleNeck(bottleneck_channels, bidirectional=bidirectional)
        
        self.decoders = nn.ModuleList()
        for idx in range(num_layers):
            if idx != num_layers - 1:
                self.decoders.append(
                    DecoderBlock(decoder_channels[idx], decoder_channels[idx+1], upscale_factor=2, is_last=False, causal=casual, use_res=False)
                )
            elif idx == 0:
                self.decoders.append(
                    DecoderBlock(decoder_channels[idx], decoder_channels[idx+1], upscale_factor=2, is_last=True, causal=casual, use_res=False)
                )
            else:
                self.decoders.append(
                    DecoderBlock(decoder_channels[idx], decoder_channels[idx+1], upscale_factor=2, is_last=True, causal=casual)
                )

        self.fc = nn.Linear(self.in_dim - 2, self.out_dim)
        self.sigmoid = nn.Sigmoid()

        self.rnnoise_module = RnnoiseModule(n_fft=512, hop_len=256, win_len=512, up_scale=64.0, nfilter=100)
            
    def transform(self, f):
        # [B, F, T] - > [B, 1, T, F]
        mag = f.real**2 + f.imag**2 + 1e-6
        pow_mag = mag ** 2
        log_pow_mag = torch.log(pow_mag)
        feat = log_pow_mag.permute(0, 2, 1).unsqueeze(1)
        return feat
    
    def forward(self, mic, far):
        wav_length = mic.shape[-1]
        out_mic = self.rnnoise_module.forward_transform(mic)
        out_ref = self.rnnoise_module.forward_transform(far)
        out_mic_cat = None

        mic_bfcc = out_mic[..., :100]
        mic_diff_1 = out_mic[:, :, :, 100:106]
        mic_diff_2 = out_mic[..., 200:206]
        ref_bfcc = out_ref[..., :100]
        ref_diff_1 = out_ref[..., 100:106]
        ref_diff_2 = out_ref[..., 200:206]

        out_mic = torch.concat([mic_bfcc, mic_diff_1, mic_diff_2], dim=-1)
        out_ref = torch.concat([ref_bfcc, ref_diff_1, ref_diff_2], dim=-1)

        # out_mic = out_mic[:, :, 1:, :]
        # out_ref = out_ref[:, :, 1:, :]

        encoder_out = []
        
        for i in range(self.num_layers):
            if i < 1:
                out_mic = self.mic_encoders[i](out_mic)
                out_ref = self.ref_encoders[i](out_ref)
                encoder_out.append(out_mic)
            elif i == 1:
                out_mic_cat = torch.cat([out_ref, out_mic], dim=1)
                out_mic_cat = self.mic_encoders[i](out_mic_cat)
                encoder_out.append(out_mic_cat)
            else:
                out_mic_cat = self.mic_encoders[i](out_mic_cat)
                encoder_out.append(out_mic_cat)
        
        feats = self.bottleneck(out_mic_cat)
        
        for i in range(self.num_layers):
            encoder_feat = encoder_out[self.num_layers-i-1]
            feats = self.decoders[i](encoder_feat, feats)

        gains = self.fc(feats)
        gains = torch.squeeze(gains, 1)
        gains = self.sigmoid(gains)

        out_specs, out_wavs, _, _ = self.rnnoise_module.inverse_transform(mic, gains)
        out_wavs = F.pad(out_wavs, (0, wav_length - out_wavs.shape[-1]))
        return {
            "gains": gains,
            "specs": out_specs,
            "wavs": out_wavs,
        }

def test_model():
    import soundfile as sf
    from pfdkf_zsm import pfdkf

    model = DeepVQES(in_dim=112, out_dim=100, casual=True, bidirectional=False)
    model.eval()
    cpt = torch.load("/home/node25_tmpdata/xcli/percepnet/train/exp/rnnvqe_v8_2/8.pt.tar", map_location="cpu")

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

    model.load_state_dict(new_state_dict)

    mic, sr = sf.read("/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/mic.wav")
    ref, sr = sf.read("/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/ref.wav")
    e, y = pfdkf(ref, mic)
    min_len = min(mic.shape[-1], ref.shape[-1], e.shape[-1], y.shape[-1])
    mic = mic[:min_len]
    ref = ref[:min_len]
    e = e[:min_len]
    y = y[:min_len]

    mic = torch.from_numpy(mic).unsqueeze(0)
    y = torch.from_numpy(y).unsqueeze(0)
    outputs = model(mic, y)["wavs"]
    print(f'output shape: {outputs.shape}')
    outputs = outputs.squeeze(0).detach().numpy()
    sf.write("/home/node25_tmpdata/xcli/percepnet/c_aec/test_wav/out_py.wav", outputs, 16000)


if __name__ == "__main__":
    test_model()
