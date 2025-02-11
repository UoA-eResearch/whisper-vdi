#!/usr/bin/env python3

import requests
import jiwer
import webvtt
import pandas as pd

reference = " ".join([caption.text for caption in webvtt.read('data/094337_tagged_20170915759438_paraini.vtt')])
results = []

def get_metrics(hypothesis, name=""):
  tr = jiwer.Compose(
      [
          jiwer.ExpandCommonEnglishContractions(),
          jiwer.RemoveEmptyStrings(),
          jiwer.ToLowerCase(),
          jiwer.RemoveMultipleSpaces(),
          jiwer.Strip(),
          jiwer.RemovePunctuation(),
          jiwer.ReduceToListOfListOfWords(),
      ]
  )
  output = jiwer.process_words(reference, hypothesis, reference_transform=tr, hypothesis_transform=tr)
  cer_output = jiwer.process_characters(reference, hypothesis, reference_transform=tr, hypothesis_transform=tr)
  return {
    "name": name,
    "wer": output.wer,
    "mer": output.mer,
    "wil": output.wil,
    "cer": cer_output.cer,
    "time": r.elapsed.total_seconds()
  }

# A100 + insanely-fast-whisper + large-v3 (Transformers) (fp16 + batching [24] + Flash Attention 2)
r = requests.post("https://a100.auckland-cer.cloud.edu.au/upload", files={"audio": open("data/094337_tagged_20170915759438_paraini.mp3", "rb").read()}, data={"language": "mi"})
hypothesis = r.json()["output"]["text"]
metrics = get_metrics(hypothesis, "A100 + insanely-fast-whisper + large-v3 (Transformers) (fp16 + batching [24] + Flash Attention 2)")
print(metrics)
results.append(metrics)

# A100 + faster-whisper + large-v3
r = requests.post("https://asr.auckland-cer.cloud.edu.au/asr", files={"audio_file": open("data/094337_tagged_20170915759438_paraini.mp3", "rb").read()}, params={"language": "mi"})
hypothesis = r.text.replace("\n", " ")
metrics = get_metrics(hypothesis, "A100 + faster-whisper + large-v3")
print(metrics)
results.append(metrics)

df = pd.DataFrame(results)
print(df)
df.to_csv("gpu_tests.csv", index=False)