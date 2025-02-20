import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = torch.finfo(torch.float32).eps

def mse(est, target):
    return F.mse_loss(est, target)

def mae(est, target):
    return F.l1_loss(est, target)

def kd_loss(inputs, labels):
    # logits kd
    # TODO: feature kd
    loss = 0.
    for s, t in zip(inputs[:2], labels[:2]):
        if torch.is_complex(s):
            s = torch.view_as_real(s)
            t = torch.view_as_real(t)
        loss += mae(s, t)
    return loss

def output_kd_loss_waveform(inputs, labels):
    if torch.is_complex(inputs):
        inputs = torch.view_as_real(inputs)
        labels = torch.view_as_real(labels)
    loss = mae(inputs, labels)
    return loss


def output_kd_loss_spec(inputs, labels, n_fft=256, win_len=240, win_hop=120):
    in_specs = torch.stft(inputs, n_fft, win_hop, win_len, window=torch.hann_window(win_len).to(inputs.device), return_complex=True)
    tg_specs = torch.stft(labels, n_fft, win_hop, win_len, window=torch.hann_window(win_len).to(labels.device), return_complex=True)

    if torch.is_complex(in_specs):
        in_specs = torch.view_as_real(in_specs)
        tg_specs = torch.view_as_real(tg_specs)
    loss = mae(in_specs, tg_specs)
    return loss

def output_kd_loss(inputs, labels):
    return output_kd_loss_waveform(inputs, labels) + output_kd_loss_spec(inputs, labels)

def get_asym_loss(est_mag, ref_mag, lamda=0.5, eps=1e-8, use_first=False, fft_cfg=None):

        press_est_mag = est_mag**lamda
        press_ref_mag = ref_mag**lamda

        asym_mag_loss = torch.square(torch.clamp(press_est_mag - press_ref_mag, max=0))

        loss = asym_mag_loss
        
        N, F, T = loss.shape
        loss = torch.mean(loss) * F
        return loss

def ccmse_loss(inputs, targets, weight_factor=0.8, compress_factor=0.3, eps=1e-6,
               fft_len=256, hop_len=120, win_len=240):
    # spectral complex compressed mean-squared error (CCMSE)
    if (inputs.shape[-1] > targets.shape[-1]):
        pad = torch.zeros(targets.shape[0], inputs.shape[-1] - targets.shape[-1], device=inputs.device)
        targets = torch.concat([targets, pad], dim=-1)
    if (inputs.shape[-1] < targets.shape[-1]):
        pad = torch.zeros(targets.shape[0], targets.shape[-1] - inputs.shape[-1], device=inputs.device)
        inputs = torch.concat([inputs, pad], dim=-1)
    # B x F x T
    cspecs_i = torch.stft(inputs, fft_len, hop_len, win_len, window=torch.hann_window(win_len).to(inputs.device), return_complex=True)
    cspecs_t = torch.stft(targets, fft_len, hop_len, win_len, window=torch.hann_window(win_len).to(targets.device), return_complex=True)

    # pha_i, pha_t = th.angle(cspecs_i), th.angle(cspecs_t)
    cspecs_i = torch.view_as_real(cspecs_i)
    real_i, imag_i = cspecs_i[...,0], cspecs_i[...,1]
    pha_i = torch.atan2(imag_i, real_i+eps)

    cspecs_t = torch.view_as_real(cspecs_t)
    real_t, imag_t = cspecs_t[...,0], cspecs_t[...,1]
    pha_t = torch.atan2(imag_t, real_t+eps)

    mag_i, mag_t = real_i**2+imag_i**2+eps, real_t**2+imag_t**2+eps
    compress_mag_i, compress_mag_t = torch.pow(mag_i, compress_factor), torch.pow(mag_t, compress_factor)    

    asym_loss = get_asym_loss(compress_mag_i, compress_mag_t)
            
    compress_cplx_i = torch.cat([compress_mag_i*torch.sin(pha_i), compress_mag_i*torch.cos(pha_i)], dim=-1)
    compress_cplx_t = torch.cat([compress_mag_t*torch.sin(pha_t), compress_mag_t*torch.cos(pha_t)], dim=-1)
    
    
    mag_loss = mse(compress_mag_i, compress_mag_t)
    cplx_loss = mse(compress_cplx_i, compress_cplx_t)  
    loss = mag_loss * weight_factor + cplx_loss * (1 - weight_factor)
    return loss, mag_loss, cplx_loss, asym_loss

def my_crossentropy(y_true, y_pred):
    if (y_pred.shape[1] < y_true.shape[1]):
        pad = torch.zeros(y_pred.shape[0], y_true.shape[1] - y_pred.shape[1], y_pred.shape[2], device=y_true.device)
        y_pred = torch.concat([y_pred, pad], dim=1)
    if (y_pred.shape[1] > y_true.shape[1]):
        pad = torch.zeros(y_true.shape[0], y_pred.shape[1] - y_true.shape[1], y_true.shape[2], device=y_true.device)
        y_true = torch.concat([y_true, pad], dim=1)
    y_true = y_true.clip(EPSILON,1.0)
    y_pred = y_pred.clip(EPSILON,1.0)
    return torch.mean(torch.mean(torch.mean(2*torch.abs(y_true-0.5) * nn.functional.binary_cross_entropy(y_pred, y_true), axis=-1), axis=-1), axis=-1)

def mymask(y_true):
    return torch.minimum(y_true+1., torch.tensor(1., device=y_true.device))

def msse(y_true, y_pred):
    return torch.mean(torch.mean(torch.mean(mymask(y_true) * torch.square(torch.sqrt(y_pred) - torch.sqrt(y_true)), axis=-1), axis=-1), axis=-1)

def mycost(y_true, y_pred):
    if (y_pred.shape[1] < y_true.shape[1]):
        pad = torch.zeros(y_pred.shape[0], y_true.shape[1] - y_pred.shape[1], y_pred.shape[2], device=y_true.device)
        y_pred = torch.concat([y_pred, pad], dim=1)
    if (y_pred.shape[1] > y_true.shape[1]):
        pad = torch.zeros(y_true.shape[0], y_pred.shape[1] - y_true.shape[1], y_true.shape[2], device=y_true.device)
        y_true = torch.concat([y_true, pad], dim=1)
    y_true = y_true.clip(EPSILON,1.0)
    y_pred = y_pred.clip(EPSILON,1.0)
    return torch.mean(
        torch.mean(
            torch.mean(
                mymask(y_true) * (10*torch.square(torch.square(torch.sqrt(y_pred) - torch.sqrt(y_true))) + torch.square(torch.sqrt(y_pred) - torch.sqrt(y_true)) + 0.01*nn.functional.binary_cross_entropy(y_pred, y_true)), axis=-1
            ), axis=-1
        ), axis=-1
    )

# def mycost(y_true, y_pred, echo_aware):
#     y_true = y_true.clip(EPSILON,1.0)
#     y_pred = y_pred.clip(EPSILON,1.0)
#     loss_matrix = mymask(y_true) * (10 * torch.square(torch.square(torch.sqrt(y_pred) - torch.sqrt(y_true))) + torch.square(torch.sqrt(y_pred) - torch.sqrt(y_true)))
#     # loss_matrix = loss_matrix * (1 + echo_aware)
#     loss = loss_matrix.mean()
#     return loss

def l2_norm(s1, s2):
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm

def si_snr(s1, s2, eps=EPSILON):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm / (s2_s2_norm + eps) *s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)

def sisnr_loss(inputs, labels):
    if (inputs.shape[-1] > labels.shape[-1]):
        pad = torch.zeros(labels.shape[0], inputs.shape[-1] - labels.shape[-1], device=inputs.device)
        labels = torch.concat([labels, pad], dim=-1)
    if (inputs.shape[-1] < labels.shape[-1]):
        pad = torch.zeros(labels.shape[0], labels.shape[-1] - inputs.shape[-1], device=inputs.device)
        inputs = torch.concat([inputs, pad], dim=-1)
    return -(si_snr(inputs, labels))

def Mag_Compress_Mse_Asym(inputs, labels, lamda=0.5, n_fft=512, hop_length=320, win_length=512, window=None):
    '''
       est_cspec: N x 2 x F x T
    '''
    if (not window is None) and window.device != inputs.device:
        window = window.to(inputs.device)

    x = torch.stft(inputs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    est_cspec = x.permute(0,3,1,2)
    x = torch.stft(labels, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    ref_cspec = x.permute(0,3,1,2)

    est_cspec = est_cspec[:, :, 1:]
    ref_cspec = ref_cspec[:, :, 1:]

    est_mag = torch.sqrt(torch.clamp(est_cspec[:,0]**2+est_cspec[:,1]**2, min=EPSILON))
    ref_mag = torch.sqrt(torch.clamp(ref_cspec[:,0]**2+ref_cspec[:,1]**2, min=EPSILON))

    press_est_mag = torch.pow(est_mag, lamda)
    press_ref_mag = torch.pow(ref_mag, lamda)

    mag_loss = torch.square(press_est_mag - press_ref_mag)
    asym_mag_loss = torch.square(torch.clamp(press_est_mag - press_ref_mag, max=0))
    loss = mag_loss + asym_mag_loss
    N, F, T = loss.shape
    loss = torch.mean(loss) * F
    return loss


def Mag_Compress_Mse(inputs, labels, lamda=0.5, n_fft=512, hop_length=320, win_length=512, window=None):
    '''
       est_cspec: N x 2 x F x T
    '''
    if (not window is None) and window.device != inputs.device:
        window = window.to(inputs.device)

    x = torch.stft(inputs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    est_cspec = x.permute(0,3,1,2)
    x = torch.stft(labels, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    ref_cspec = x.permute(0,3,1,2)

    est_cspec = est_cspec[:, :, 1:]
    ref_cspec = ref_cspec[:, :, 1:]

    est_mag = torch.sqrt(torch.clamp(est_cspec[:,0]**2+est_cspec[:,1]**2, min=EPSILON))
    ref_mag = torch.sqrt(torch.clamp(ref_cspec[:,0]**2+ref_cspec[:,1]**2, min=EPSILON))

    press_est_mag = torch.pow(est_mag, lamda)
    press_ref_mag = torch.pow(ref_mag, lamda)

    mag_loss = torch.square(press_est_mag - press_ref_mag)
    loss = mag_loss
    N, F, T = loss.shape
    loss = torch.mean(loss) * F
    return loss


if __name__ == "__main__":
    a = torch.rand(4, 102, 100)
    b = torch.rand(4, 104, 100)
    print(my_crossentropy(a,b))
    print(mycost(a,b))
    # print(sisnr_loss(a, b))
    # print(ccmse_loss(a, b))
    # print(msse(a,b))