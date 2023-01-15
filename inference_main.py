import argparse

import parselmouth
import soundfile
import librosa
import torch
import numpy as np

import cluster
import utils
import json

from inference.infer_tool import Svc
import logging
logging.getLogger('numba').setLevel(logging.WARNING)

sample_rate = 44100
hop_len = 512


def getc(audio_data, hmodel, spk):
    devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = audio_data
    wav = librosa.resample(wav,sr, 16000)
    wav = torch.from_numpy(wav).unsqueeze(0).to(devive)
    c = utils.get_vec_units(hmodel, wav).cpu().squeeze(0)
    c = utils.repeat_expand_2d(c, int((wav.shape[1] * sample_rate / 16000) // hop_len)).numpy()
    c = cluster.get_cluster_center_result(c.transpose(), spk)
    return c

def get_f0(audio_data, p_len=None, f0_up_key=0):
    x, sr = audio_data
    x = librosa.resample(x,sr, sample_rate)
    if p_len is None:
        p_len = x.shape[0] // hop_len
    else:
        assert abs(p_len - x.shape[0] // hop_len) < 3, (p_len, x.shape)
    time_step = hop_len / sample_rate * 1000
    f0_min = 50
    f0_max = 1100
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    f0 = parselmouth.Sound(x, sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']

    pad_size = (p_len - len(f0) + 1) // 2
    if (pad_size > 0 or p_len - len(f0) - pad_size > 0):
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode='constant')

    f0 *= pow(2, f0_up_key / 12)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    f0_coarse = np.rint(f0_mel).astype(np.int)
    return f0_coarse, f0

hmodel = utils.load_vec_model()


def infer(spk, raw, model, trans=0):
    audio_data = librosa.load(raw, sr=None)
    c = getc(audio_data, hmodel, spk)
    _, f0 = get_f0(audio_data, c.shape[0], trans)
    # tgt = raw.replace("raw", "results").replace(".wav", f'_{spk}_{trans}_{step}step.wav')
    c = torch.from_numpy(c).float().unsqueeze(0).transpose(1, 2)
    f0 = torch.from_numpy(f0).float().unsqueeze(0)
    res = model.infer(spk, c, f0)
    # soundfile.write(tgt, res[0].cpu().numpy(), 44100)
    return res


if __name__ == '__main__':
    step = 34000
    model_path = f"logs/G_{step}.pth"
    config_path = "configs/config.json"
    svc_model = Svc(model_path, config_path)
    raw = 'raw/君の知らない物語.wav'
    spk = "taffy"

    infer(spk, raw, svc_model)
