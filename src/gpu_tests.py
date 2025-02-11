#!/usr/bin/env python3

import requests
import pandas as pd

results = []

def save(hypothesis, name=""):
  with open(f"data/{name}.txt", 'w') as f:
      f.write(hypothesis)

# A100 + insanely-fast-whisper + large-v3 (Transformers) (fp16 + batching [24] + Flash Attention 2)
r = requests.post("https://a100.auckland-cer.cloud.edu.au/upload", files={"audio": open("data/094337_tagged_20170915759438_paraini.mp3", "rb").read()})#, data={"language": "mi"})
hypothesis = r.json()["output"]["text"]
name = "A100 + insanely-fast-whisper + large-v3 (Transformers) (fp16 + batching [24] + Flash Attention 2)"
save(hypothesis, name)
results.append({
    "name": name,
    "time": r.elapsed.total_seconds()
})

# A100 + faster-whisper + large-v3
r = requests.post("https://asr.auckland-cer.cloud.edu.au/asr", files={"audio_file": open("data/094337_tagged_20170915759438_paraini.mp3", "rb").read()})#, params={"language": "mi"})
hypothesis = r.text.replace("\n", " ")
name = "A100 + faster-whisper + large-v3"
metrics = save(hypothesis, name)
results.append({
    "name": name,
    "time": r.elapsed.total_seconds()
})

df = pd.DataFrame(results)
print(df)
df.to_csv("gpu_tests.csv", index=False)