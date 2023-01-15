import librosa
import numpy as np
import parselmouth
import pyworld
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
from inference.infer_tool import Svc
from mel_processing import spectrogram_torch
from models import SynthesizerTrn
from utils import load_wav_to_torch
import utils
import torch
import soundfile
spk_dict ={
    "taffy": 0,
    "nyaru": 1
  }
model_path = "/Volumes/Extend/AI/nyaru3.1/logs/32k/NyaruTaffy.pth"
config_path = "configs/nyarutaffy.json"

hps_ms = utils.get_hparams_from_file(config_path)

srcpath = "raw/000009.wav"
audio, sr = librosa.load(srcpath, 32000)
soundfile.write(srcpath, audio, sr)
audio, sampling_rate = load_wav_to_torch(srcpath)

y = audio / hps_ms.data.max_wav_value
y = y.unsqueeze(0)

spec = spectrogram_torch(y, hps_ms.data.filter_length,
                         hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                         center=False)

spec_lengths = torch.LongTensor([spec.size(-1)])
sid_src = torch.LongTensor([0])


net_g_ms = SynthesizerTrn(
                hps_ms.data.filter_length // 2 + 1,
                hps_ms.train.segment_size // hps_ms.data.hop_length,
                **hps_ms.model)
_ = net_g_ms.eval()
_ = utils.load_checkpoint(model_path, net_g_ms, None)



svc_model = Svc(model_path, config_path)
def resize2d_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res

def compute_f0(path, c_len):
    x, sr = librosa.load(path, sr=32000)
    f0, t = pyworld.dio(
        x.astype(np.double),
        fs=sr,
        f0_ceil=800,
        frame_period=1000 * 320 / sr,
    )
    f0 = pyworld.stonemask(x.astype(np.double), f0, t, 32000)
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)

    return None, resize2d_f0(f0, c_len)


_, f0 = compute_f0(srcpath, spec.shape[-1])
f0 = torch.FloatTensor(f0).unsqueeze(0)
sid_tgt = torch.LongTensor([1])

audio1 = net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt, f0=f0)[0][0, 0].data.float().numpy()
soundfile.write("out.wav", audio1, 32000)
print(f0.shape)



