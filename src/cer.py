#!/usr/bin/env python3

import jiwer
import webvtt
import pandas as pd
from torchmetrics.text import CharErrorRate
from glob import glob
import os
from tqdm.auto import tqdm

def get_metrics(reference, hypothesis, name=""):
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
    "wer": output.wer,
    "mer": output.mer,
    "wil": output.wil,
    "cer": cer,
    "untransformed_cer": untransformed_cer,
  }

results = []
df = pd.read_csv("gpu_tests.csv")
for i, row in tqdm(df.iterrows(), total=len(df)):
    # name,file,file_duration,time,time_per_second,transcript
    try:
      filename = "data/" + row.file.replace(".mp3", ".vtt")
      reference = " ".join([caption.text for caption in webvtt.read(filename)])
    except:
      try:
        reference = open("data/" + row.file.replace(".mp3", ".txt")).read()
      except:
        continue
    metrics = get_metrics(row.transcript, reference)
    row = row.to_dict()
    row.update(metrics)
    results.append(row)

df = pd.DataFrame(results).drop(columns="transcript").sort_values("time_per_second")
print(df)
df.to_csv("gpu_results.csv", index=False)
