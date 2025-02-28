import pywhispercpp
from pywhispercpp.model import Model
#from pywhispercpp.progress import whisper_progress_callback
from tqdm import tqdm
import os
import time
import numpy as np

import jiwer
import webvtt
import pandas as pd

def get_metrics(reference, hypothesis, name=""):
    #reference = open("data/paraini.txt", "rt").read()
    tr = jiwer.Compose(
        [
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    output = jiwer.process_words(reference, hypothesis, reference_transform=tr, hypothesis_transform=tr)
    cer = jiwer.cer(reference, hypothesis, reference_transform=tr, hypothesis_transform=tr)
    untransformed_cer = jiwer.cer(reference.lower(), hypothesis.lower())
    return {
    "name": name,
    "wer": output.wer,
    "mer": output.mer,
    "wil": output.wil,
    "cer": cer,
    "untransformed_cer": untransformed_cer,
    }

pbar = tqdm(initial=0, total=100)
def callback(ctx, state, i, p):
    print(f"{ctx}, {state}, {i}, {p}")
    #pbar.update(i)

path = '/mnt/whisper-vdi/data/'
audio_file = path + "cerebral.mp3"
num_threads = os.cpu_count()

audio_fname = audio_file.removeprefix(path)
audio_fname = audio_fname.split('.')[0]

start = time.time()

model_names = ['large-v3-turbo', 'large-v3']
device = 'cpu'
batch_size = 1
beam_size = 1

for model_name in model_names:
    #model = Model(model_path="/mnt/whisper-vdi/models/ggml-large-v3-turbo.bin", n_threads=num_threads)
    model = Model(model_name, models_dir='/mnt/whisper-vdi/models/', n_threads = 4)
    #model.params.progress_callback = whisper_progress_callback(callback)
    segments = model.transcribe(audio_file)

    with open(path + 'whisper-cpp-' + model_name + '_' + audio_fname +'.txt', 'w') as f:
        for segment in segments:
            f.write(segment.text)
        f.close()

    end = time.time()
    runtime = end - start
    print(runtime)

    if not os.path.exists(path + 'wcpp_preds_' + audio_fname + '.csv'):
        heads = ['Model', 'Device', 'num_threads', 'runtime', 'batch_size', 
                'beam_size', "wer", "mer", "wil", "cer", "untransformed_cer"]
        heads = np.reshape(np.array(heads), (1, -1))
        df = pd.DataFrame(data=heads)
        df.to_csv(path + 'wcpp_preds_' + audio_fname + '.csv', mode='a', header=False, index=False)

    ref = open(path + audio_fname + ".txt", "rt").read()
    hyp = open(path + 'whisper-cpp-' + model_name + '_' + audio_fname +'.txt', "rt").read()

    metrics = get_metrics(ref, hyp)

    df_data=[model_name, device, num_threads, runtime, batch_size, 
                beam_size, metrics["wer"], metrics["mer"], metrics["wil"], metrics["cer"], metrics["untransformed_cer"]]
    df_data = np.reshape(np.array(df_data), (1, -1))
    df = pd.DataFrame(data=df_data)
    df.to_csv(path + 'wcpp_preds_' + audio_fname + '.csv', mode='a', header=False, index=False)