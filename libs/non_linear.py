
import numpy as np
from scipy import integrate
import scipy.signal as ss


def nonlinear1(sig):
    # must with reverb
    def f(x, eps_2=0.1):
        return np.exp(-x**2/(2*eps_2))
    nl_sig = np.empty_like(sig)
    for i in range(len(sig)):
        output = integrate.quad(f, 0, sig[i])[0]
        nl_sig[i] = output
    return nl_sig


def nonlinear2(sig):
    # must with reverb
    max_amp = np.max(np.abs(sig))
    sig_hard = np.clip(sig, -max_amp*0.8, max_amp*0.8)
    bn = 1.5*sig_hard-0.3*sig_hard**2
    alpha = (bn > 0)*3.5 + 0.5
    omega = 4
    sig_non_linear = omega*(2/(1+np.exp(-alpha*bn))-1)
    return sig_non_linear


def nonlinear3(sig):
    max_norm = np.random.uniform(0.8, 1.0)
    sig = np.clip(sig, -max_norm, max_norm)
    return sig


def echoclip(sig):
    sig = 5*sig
    sig = np.clip(sig, -1, 1)
    return sig



def reverb(signal, rir):
    # 50ms reverb
    early_rir = rir[:800]
    if signal.ndim != rir.ndim:
        print(signal.ndim)
        print(rir.ndim)
        print(f'signal shape: {signal.shape}')
        print(f'rir shape: {rir.shape}')
    late_reverb = ss.oaconvolve(signal, rir)[:len(signal)]
    early_reverb = ss.oaconvolve(signal, early_rir)[:len(signal)]
    return early_reverb, late_reverb
