import io
import  os
os.system("wget -P cvec/ https://huggingface.co/spaces/innnky/nanami/resolve/main/checkpoint_best_legacy_500.pt")
import gradio as gr
import librosa
import numpy as np
import soundfile
import torch
from inference.infer_tool import Svc
import inference_main
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

config_path = "configs/config.json"

model_34k = Svc("logs/G_34000.pth", config_path)
model_139k = Svc("logs/G_139000.pth", config_path)

model_map = {
    "G_34000.pth": model_34k,
    "G_139000.pth": model_139k
}
def vc_fn(sid, input_audio, vc_transform, model):
    if input_audio is None:
        return "You need to upload an audio", None
    sampling_rate, audio = input_audio
    # print(audio.shape,sampling_rate)
    duration = audio.shape[0] / sampling_rate
    if duration > 45:
        return "请上传小于45s的音频，需要转换长音频请本地进行转换", None
    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
    print(audio.shape)
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, audio, 16000, format="wav")
    out_wav_path.seek(0)

    out_audio, out_sr = inference_main.infer(sid, out_wav_path, model_map[model], vc_transform)
    _audio = out_audio.cpu().numpy()
    return "Success", (44100, _audio)


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("Basic"):
            gr.Markdown(value="""
                这是ai猫雷3.5版本demo，算是一个不太一样的尝试，图一乐就行

                模型采用了聚类的方案对content vec进行离散化，主要是针对于解决猫雷模型不够"像"猫雷的问题
                
                牺牲了部分咬字性能（可能会有很多发音错误），但是会更加像目标音色（大概？

                暂时不提供训练代码

                本地合成可以删除32、33两行代码以解除合成45s长度限制""")
            sid = gr.Dropdown(label="音色", choices=['nyaru', "taffy", "otto"], value="nyaru")
            vc_input3 = gr.Audio(label="上传音频（长度小于45秒）")
            vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
            model = gr.Dropdown(label="模型", choices=list(model_map.keys()), value="G_34000.pth")
            vc_submit = gr.Button("转换", variant="primary")
            vc_output1 = gr.Textbox(label="Output Message")
            vc_output2 = gr.Audio(label="Output Audio")
        vc_submit.click(vc_fn, [sid, vc_input3, vc_transform, model], [vc_output1, vc_output2])

    app.launch(server_port=7860)



