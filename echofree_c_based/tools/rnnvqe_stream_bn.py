import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from functools import partial
import numpy as np

import sys, os
sys.path.append(os.path.dirname(__file__))
from tfilm import Frequency_FiLM

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(project_root)

from pyrnnoise_stream.rnnoise_module_stream import RnnoiseModule
from nnet.norm import new_norm
from linear_model.pfdkf_zsm import pfdkf

class StreamingDepthwiseConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: int,
                 padding: tuple):
        super().__init__()
        # Set padding to zero in the time dimension
        padding = (0, padding[1])
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.kernel_size = kernel_size
        self.buffer = [None]

    def forward(self, x):
        # x: [B, C, T, F]
        buffer_size = self.kernel_size[0] - 1
        if self.buffer[0] is None:
            self.buffer[0] = torch.zeros(
                x.shape[0], x.shape[1], buffer_size, x.shape[3],
                device=x.device, dtype=x.dtype)
        x_padded = torch.cat([self.buffer[0], x], dim=2)
        x = self.depth_conv(x_padded)
        self.buffer[0] = x_padded[:, :, -buffer_size:, :]
        x = self.point_conv(x)
        return x

    def reset_buffer(self):
        self.buffer = [None]

class ResidualBlock(nn.Module):
    def __init__(self,
                hidden_channels: int,
                kernel_size: tuple = (4, 3),
                stride: int = 1,
                causal: bool = True,
                ):
        super().__init__()
        # p_time = (kernel_size[0] - 1) 
        self.causal = causal
        self.kernel_size = kernel_size
        self.buffer = [None]
        p_freq = (kernel_size[1] - 1) // 2
        self.conv2d = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, (0, p_freq))
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.act = nn.ELU()
    
    def forward(self, x):
        # x in shape of [Batch, Channel, Time, Frequency] 
        buffer_size = self.kernel_size[0] - 1
        if self.buffer[0] is None:
            self.buffer[0] = torch.zeros([x.shape[0], x.shape[1], buffer_size, x.shape[3]], device=x.device, dtype=x.dtype)
        padded_x = torch.concat([self.buffer[0], x], dim=2)
        self.buffer[0] = padded_x[:, :, -buffer_size:, :]
        # print(padded_x.shape)
        z = self.conv2d(padded_x)
        # print(z.shape)
        # if self.causal:
        #     z = z[:,:,:-buffer_size,:]
        z = self.bn(z)
        z = self.act(z)
        y = z + x 
        return y
    
    def reset_buffer(self):
        self.buffer = [None]

class StreamingEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_res: bool=True, 
                 kernel_size: tuple = (4, 3),
                 stride: tuple = (1, 2),
                 causal: bool = True,
                 ):
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.conv2d = StreamingDepthwiseConv2d(in_channels, out_channels, kernel_size, stride, (0, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()
        self.use_res = use_res
        if self.use_res:
            self.resblock = ResidualBlock(out_channels, causal=causal)

    def forward(self, x):
        z = self.conv2d(x)
        z = self.bn(z)
        z = self.act(z)
        if self.use_res:
            z = self.resblock(z)
        return z

    def reset_buffer(self):
        self.conv2d.reset_buffer()
        if self.use_res:
            self.resblock.reset_buffer()

class StreamingBottleNeck(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional
        if bidirectional:
            self.rnn = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_state = [None]

    def forward(self, x):
        # x in shape of [Batch, Channel, Time, Frequency]
        x = x.permute(0, 2, 1, 3)
        B, T, C, F = x.shape
        x = x.reshape(B, T, -1).contiguous()
        x, self.hidden_state[0] = self.rnn(x, self.hidden_state[0])
        # x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.reshape(B, T, C, F).permute(0, 2, 1, 3)
        return x

    def reset_hidden_state(self):
        self.hidden_state = [None]

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

class StreamingDecoderBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 upscale_factor: int = 2,
                 is_last: bool = False,
                 causal: bool = True,
                 use_res: bool = True,
                 ):
        super().__init__()        
        self.skip = SkipBlock(in_channels)
        self.subpixconv = SubPixelConv(in_channels, out_channels, upscale_factor)
        self.use_res = use_res
        if self.use_res:
            self.resblock = ResidualBlock(in_channels, causal=causal)     
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
    
    def reset_buffer(self):
        if self.use_res:
            self.resblock.reset_buffer()

class DeepVQES(nn.Module):
    '''Streamable version of DeepVQES'''
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
                    StreamingEncoderBlock(encoder_mic_channels[idx], encoder_mic_channels[idx + 1], causal=casual, use_res=False)
                )
                self.ref_encoders.append(
                    StreamingEncoderBlock(encoder_ref_channels[idx], encoder_ref_channels[idx + 1], causal=casual, use_res=False)
                )
            elif idx == 1:
                self.mic_encoders.append(
                    StreamingEncoderBlock(encoder_mic_channels[idx] + encoder_ref_channels[idx], encoder_mic_channels[idx + 1], causal=casual, use_res=False)
                )
            else:
                self.mic_encoders.append(
                    StreamingEncoderBlock(encoder_mic_channels[idx], encoder_mic_channels[idx + 1], causal=casual, use_res=False)
                )

        self.bottleneck = StreamingBottleNeck(bottleneck_channels, bidirectional=bidirectional)

        # Decoder remains the same; you may need to modify it similarly if it uses convolution layers
        self.decoders = nn.ModuleList()
        for idx in range(num_layers):
            if idx != num_layers - 1:
                self.decoders.append(
                    StreamingDecoderBlock(decoder_channels[idx], decoder_channels[idx + 1], upscale_factor=2, is_last=False, causal=casual, use_res=False)
                )
            elif idx == 0:
                self.decoders.append(
                    StreamingDecoderBlock(decoder_channels[idx], decoder_channels[idx + 1], upscale_factor=2, is_last=True, causal=casual, use_res=False)
                )
            else:
                self.decoders.append(
                    StreamingDecoderBlock(decoder_channels[idx], decoder_channels[idx + 1], upscale_factor=2, is_last=True, causal=casual, use_res=True)
                )

        self.fc = nn.Linear(self.in_dim - 2, self.out_dim)
        self.sigmoid = nn.Sigmoid()

        self.rnnoise_module = RnnoiseModule(n_fft=512, hop_len=256, win_len=512, up_scale=64.0, nfilter=100)

    def reset_buffers(self):
        self.rnnoise_module.reset_buffer()
        for layer in self.mic_encoders:
            layer.reset_buffer()
        for layer in self.ref_encoders:
            layer.reset_buffer()
        for layer in self.decoders:
            layer.reset_buffer()
        self.bottleneck.reset_hidden_state()

    def forward(self, mic, far):
        wav_length = mic.shape[-1]
        if len(mic.shape) == 1:
            mic = mic.unsqueeze(0)
            far = far.unsqueeze(0)
        out_mic = self.rnnoise_module.forward_transform(mic, 0)
        out_ref = self.rnnoise_module.forward_transform(far, 1)
        feats_out = out_mic.clone().detach()
        out_mic_cat = None

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
        
        feats_out = out_mic_cat.clone().detach()

        feats = self.bottleneck(out_mic_cat)

        for i in range(self.num_layers):
            encoder_feat = encoder_out[self.num_layers - i - 1]
            feats = self.decoders[i](encoder_feat, feats)

        gains = self.fc(feats)
        gains = torch.squeeze(gains, 1)
        gains = self.sigmoid(gains)

        out_specs, out_wavs = self.rnnoise_module.inverse_transform(mic, gains)
        # out_wavs = F.pad(out_wavs, (0, wav_length - out_wavs.shape[-1]))
        return {
            "gains": gains,
            "specs": out_specs,
            "wavs": out_wavs,
            "feats": feats_out
        }

def vorbis_window(n):
    """Generate a Vorbis window of length n."""
    window = np.zeros(n)
    for i in range(n):
        window[i] = np.sin((np.pi / 2) * (np.sin(np.pi * (i + 0.5) / n)) ** 2)
    return torch.from_numpy(window.astype(np.float32))

def test_model():
    import torch
    import soundfile as sf
    import numpy as np
    import librosa
    
    model = DeepVQES(in_dim=112, out_dim=100, casual=True, bidirectional=False)

    # cpt = torch.load("/home/node25_tmpdata/xcli/percepnet/train/exp/rnnvqe_v8_2/8.pt.tar", map_location="cpu")
    cpt = torch.load("/home/node25_tmpdata/xcli/percepnet/upload_scripts/rnnvqe_v8_best.pt.tar", map_location="cpu")
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
    model.eval()

    start = 0
    end = 40

    mic, sr = sf.read("/home/node25_tmpdata/xcli/percepnet/train/FBMicRef/安静2_mic.wav")
    ref, sr = sf.read("/home/node25_tmpdata/xcli/percepnet/train/FBMicRef/安静2_lpb.wav")
    mic = librosa.resample(mic, orig_sr=sr, target_sr=16000)[start*16000:end*16000]
    ref = librosa.resample(ref, orig_sr=sr, target_sr=16000)[start*16000:end*16000]
    laec_outputs, laec_echo = pfdkf(ref, mic)

    min_len = min(laec_outputs.shape[0], ref.shape[0])
    mic = mic[:min_len]
    laec_outputs = laec_outputs[:min_len]
    laec_echo = laec_echo[:min_len]
    ref = ref[:min_len]

    mic = torch.from_numpy(mic)
    laec_echo = torch.from_numpy(laec_echo)

    print(mic.shape)
    outputs_ori = model(mic, laec_echo)["wavs"]
    print(f'output_ori: {outputs_ori}')
    outputs_ori = outputs_ori.squeeze(0).detach().numpy()
    model.reset_buffers()

    length = mic.shape[-1]
    start = 0
    win_len = 512
    hop_len = 256
    outputs = 0
    
    while (start + win_len <= length):
        inputs_frame = mic[start:start+win_len]
        laec_echo_frame = laec_echo[start:start+win_len]
        start = start + hop_len
        outputs_frame = model(inputs_frame, laec_echo_frame)["specs"]
        if start == hop_len:
            outputs = outputs_frame
        else:
            outputs = torch.concat([outputs, outputs_frame], dim=-2)
            # zero_pad = torch.zeros(outputs.shape[0], hop_len, device=outputs.device)
            # outputs = torch.concat([outputs, zero_pad], dim=-1)
            # outputs[:, -win_len:] += outputs_frame
    window = vorbis_window(512)
    outputs_stream = torch.istft(outputs.permute(0, 2, 1), n_fft=512, hop_length=256, win_length=512, window=window, return_complex=False, center=False)
    # print(f'outputs: {outputs}')
    outputs_stream = outputs_stream.squeeze(0).detach().numpy()
    
    sf.write("/home/node25_tmpdata/xcli/percepnet/train/test.wav", outputs_ori, 16000)
    sf.write("/home/node25_tmpdata/xcli/percepnet/train/test_stream.wav", outputs_stream, 16000)

# Test the modified model
def test_model_stream():
    import torch
    from tqdm import tqdm
    import soundfile as sf
    import librosa
    import numpy as np
    from pathlib import Path

    model = DeepVQES(in_dim=112, out_dim=100, casual=True, bidirectional=False)
    model.eval()

    inputs = torch.randn(1, 512 + 256 * 10)
    outputs_ori = model(inputs, inputs)["gains"]
    print(outputs_ori.shape)
    print(outputs_ori)

    model.reset_buffers()

    outputs = 0
    start = 0
    win_len = 512
    hop_len = 256
    length = inputs.shape[-1]
    
    while (start + win_len <= length):
        inputs_frame = inputs[:, start:start+win_len]
        start = start + hop_len
        outputs_frame = model(inputs_frame, inputs_frame)["gains"]
        if start == hop_len:
            outputs = outputs_frame
        else:
            zero_pad = torch.zeros(outputs.shape[0], 1, outputs.shape[-1], device=outputs.device)
            outputs = torch.concat([outputs, zero_pad], dim=1)
            outputs[:, -1:, :] += outputs_frame
            # zero_pad = torch.zeros(outputs.shape[0], hop_len, device=outputs.device)
            # outputs = torch.concat([outputs, zero_pad], dim=-1)
            # outputs[:, -win_len:] += outputs_frame
    print(outputs.shape)
    print(outputs)
    torch.set_printoptions(threshold=torch.inf)
    print((outputs_ori - outputs)[:300])
    print(torch.max((outputs_ori - outputs) / outputs_ori))

def test_cnn_stream():
    import torch
    import numpy
    torch.set_printoptions(threshold=torch.inf)
    cnn = StreamingEncoderBlock(1, 1, use_res=True, causal=True)
    cnn.eval()
    inputs = torch.randn(1, 1, 10, 112)   # [B, C, T, F]
    outputs_ori = cnn(inputs)
    cnn.reset_buffer()

    outputs_stream = 0
    for i in range(inputs.shape[2]):
        outputs_frame = cnn(inputs[:, :, i:i+1, :])
        if i == 0:
            outputs_stream = outputs_frame
        else:
            outputs_stream = torch.concat([outputs_stream, outputs_frame], dim=2)
    print((outputs_ori - outputs_stream) / outputs_ori)
    print(torch.max((outputs_ori - outputs_stream) / outputs_ori))

def test_rnn_stream():
    import torch
    import numpy
    rnn = StreamingBottleNeck(10, bidirectional=False)
    rnn.eval()

    inputs = torch.randn(1, 1, 100, 10)   # [B, C, T, F]
    outputs_ori = rnn(inputs)
    rnn.reset_hidden_state()

    print(outputs_ori.shape)
    print(outputs_ori)

    outputs_stream = 0
    for i in range(inputs.shape[2]):
        outputs_frame = rnn(inputs[:, :, i:i+1, :])
        if i == 0:
            outputs_stream = outputs_frame
        else:
            outputs_stream = torch.concat([outputs_stream, outputs_frame], dim=2)
    
    print(outputs_stream.shape)
    print(outputs_stream)
    print(torch.max((outputs_ori - outputs_stream) / outputs_ori))

def test_causal():
    import torch
    import numpy as np
    model = DeepVQES(in_dim=112, out_dim=100, casual=True, bidirectional=False)
    model.eval()
    inputs = torch.randn(3, 16000)
    outputs_1 = model(inputs, inputs)["wavs"]
    model.reset_buffers()
    outputs_2 = model(inputs[:, :10000], inputs[:, :10000])["wavs"]
    print(outputs_1[:10000])
    print(outputs_2)

def test_resblock():
    import torch
    import numpy
    torch.set_printoptions(threshold=torch.inf)
    cnn = ResidualBlock(hidden_channels=10, causal=True)
    cnn.eval()
    inputs = torch.randn(1, 10, 100, 112)   # [B, C, T, F]
    outputs_ori = cnn(inputs)
    print(outputs_ori.shape)
    print(outputs_ori)
    cnn.reset_buffer()

    outputs_stream = 0
    for i in range(inputs.shape[2]):
        outputs_frame = cnn(inputs[:, :, i:i+1, :])
        if i == 0:
            outputs_stream = outputs_frame
        else:
            outputs_stream = torch.concat([outputs_stream, outputs_frame], dim=2)
    print(outputs_stream.shape)
    print(outputs_stream)
    print((outputs_ori - outputs_stream) / outputs_ori)
    print(torch.max((outputs_ori - outputs_stream) / outputs_ori))

if __name__ == "__main__":
    # test_causal()
    test_model()
    # test_model_stream()
    # test_cnn_stream()
    # test_resblock()
    # test_rnn_stream()
    # import soundfile as sf
    # outputs_ori, sr = sf.read("/home/node25_tmpdata/xcli/percepnet/train/test.wav")
    # outputs_stream, sr = sf.read("/home/node25_tmpdata/xcli/percepnet/train/test_stream.wav")
    # print(outputs_ori[:1000])
    # print(outputs_ori.shape)
    # print(outputs_stream[:1000])
    # print(outputs_ori[:1000] - outputs_stream[:1000])