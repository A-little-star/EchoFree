import os
import json
import logging
import numpy as np


def detect_nan(input):
    num_nan = np.sum(np.isnan(input) == True)
    return num_nan > 0


def parse_scp(lst_pth):
    with open(lst_pth) as fid:
        lines = fid.readlines()
    res = []
    for i in range(len(lines)):
        tmp = lines[i].strip()
        res.append(tmp)
    return res


def delay(echo_signal, delay_ms=512):
    delay_ms = np.random.uniform(10, delay_ms)
    paddings = int(delay_ms*16)
    echo_signal = np.pad(echo_signal, (paddings, 0))[:-paddings]
    return echo_signal


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Get python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def dump_json(obj, fdir, name):
    """
    Dump python object in json
    """
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(os.path.join(fdir, name), "w") as f:
        json.dump(obj, f, indent=4, sort_keys=False)


def load_json(fdir, name):
    """
    Load json as python object
    """
    path = os.path.join(fdir, name)
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj


def tail_to_len(wave, t_len):
    while len(wave) < t_len:
        wave = np.tile(wave, 2)
    wave = wave[0:t_len]
    return wave


def trunc_to_len(wave1, t_len, prob=1):
    ''' with prob add 0, else tile
    '''
    wave_len = len(wave1)
    assert wave_len > 0
    if wave_len >= t_len:
        random_s = np.random.randint(wave_len - t_len + 1)
        wave1 = wave1[random_s:random_s + t_len]
        return wave1

    if np.random.uniform(0, 1) < prob:
        len_diff = t_len - wave_len
        random_s = np.random.randint(len_diff)
        wave2 = np.zeros(t_len)
        wave2[random_s:random_s + wave_len] = wave1
        return wave2
    else:
        while len(wave1) < t_len:
            wave1 = np.tile(wave1, 2)
        wave1 = wave1[0:t_len]
        return wave1
