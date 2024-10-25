import json
import math
import os
import time

import IPython.display as ipd
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from scipy.io.wavfile import write
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, T5ForConditionalGeneration
from underthesea import word_tokenize

import commons
import utils
from data_utils import (
    TextAudioCollate,
    TextAudioLoader,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from models import SynthesizerTrn
from text2phonemesequence import Text2PhonemeSequence
from vinorm import TTSnorm


def get_inputs(
    text, model: Text2PhonemeSequence, tokenizer_xphonebert: PreTrainedTokenizer
):
    phones = model.infer_sentence(text)
    tokenized_text = tokenizer_xphonebert(phones)
    input_ids = tokenized_text["input_ids"]
    attention_mask = tokenized_text["attention_mask"]
    input_ids = torch.LongTensor(input_ids).to("cuda:1")
    attention_mask = torch.LongTensor(attention_mask).to("cuda:1")
    return input_ids, attention_mask


hps = utils.get_hparams_from_file("configs/infore_25h_base_xphonebert.json")
tokenizer_xphonebert = AutoTokenizer.from_pretrained(hps.bert)
# Load Text2PhonemeSequence
model = Text2PhonemeSequence(language="vie-n", device="cuda:1")
net_g = SynthesizerTrn(
    hps.bert,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
).to("cuda:1")
_ = net_g.eval()

text = "Lê Quý Đôn, tên thuở nhỏ là Lê Danh Phương, là vị quan thời Lê trung hưng, cũng là nhà thơ và được mệnh danh là nhà bác học lớn của Việt Nam trong thời phong kiến."
norm_text = TTSnorm(text)
text_segment = word_tokenize(norm_text, format="text")
print(text_segment)

_ = utils.load_checkpoint("logs/infore_25h_p2//G_1220000.pth", net_g, None)

# text_segment = "hiện_nay vị_trí của bàn_thờ thường được đặt trong phòng riêng ở tầng trên cùng của nhà ."

start = time.time()
stn_tst, attention_mask = get_inputs(text_segment, model, tokenizer_xphonebert)
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0).to("cuda:1")
    attention_mask = attention_mask.to("cuda:1").unsqueeze(0)

    audio = net_g.infer(
            x_tst,
            attention_mask,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1,
        )[0][0, 0].data.cpu().float().numpy()
    
print("Time: ", time.time() - start)

sf.write('output.wav', audio, hps.data.sampling_rate)

