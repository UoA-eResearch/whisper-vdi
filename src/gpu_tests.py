#!/usr/bin/env python3

import requests
import pandas as pd
from glob import glob
import librosa
from tqdm.auto import tqdm
import os

df = pd.DataFrame({"filename": glob("data/*.mp3")})
df["duration"] = df["filename"].apply(lambda f: librosa.get_duration(path=f))
df.drop_duplicates("duration", inplace=True)
print(df)
results = []

for row in tqdm(df.itertuples(index=False), total=len(df)):
    # A100 + insanely-fast-whisper + large-v3 (Transformers) (fp16 + batching [24] + Flash Attention 2)
    r = requests.post("https://a100.auckland-cer.cloud.edu.au/upload", files={"audio": open(row.filename, "rb").read()})#, data={"language": "mi"})
    hypothesis = r.json()["output"]["text"]
    name = "A100 + insanely-fast-whisper + large-v3 (Transformers) (fp16 + batching [24] + Flash Attention 2)"
    results.append({
        "name": name,
        "file": os.path.basename(row.filename),
        "file_duration": row.duration,
        "time": r.elapsed.total_seconds(),
        "time_per_second": r.elapsed.total_seconds() / row.duration,
        "transcript": hypothesis
    })

print(pd.DataFrame(results))
pd.DataFrame(results).to_csv("gpu_tests.csv", index=False)

for row in tqdm(df.itertuples(index=False), total=len(df)):
    # A100 + faster-whisper + large-v3
    r = requests.post("https://asr.auckland-cer.cloud.edu.au/asr", files={"audio_file": open(row.filename, "rb").read()})#, params={"language": "mi"})
    hypothesis = r.text.replace("\n", " ")
    name = "A100 + faster-whisper + large-v3"
    results.append({
        "name": name,
        "file": os.path.basename(row.filename),
        "file_duration": row.duration,
        "time": r.elapsed.total_seconds(),
        "time_per_second": r.elapsed.total_seconds() / row.duration,
        "transcript": hypothesis
    })

print(pd.DataFrame(results))
pd.DataFrame(results).to_csv("gpu_tests.csv", index=False)