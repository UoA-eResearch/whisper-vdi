#!/usr/bin/env python3

import jiwer
import webvtt
import pandas as pd
from torchmetrics.text import CharErrorRate
from glob import glob
import os
from tqdm.auto import tqdm

reference = open("data/paraini.txt", "rt").read()

def get_metrics(hypothesis, name=""):
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

results = []
for f in tqdm(glob("data/*.txt")):
    hypothesis = open(f, "rt").read()
    name = os.path.basename(f).replace(".txt", "")
    metrics = get_metrics(hypothesis, name)
    results.append(metrics)

df = pd.DataFrame(results)

timings = pd.concat((pd.read_csv("gpu_tests.csv"), pd.read_csv("data/transcription_times_paraini.csv")[["name", "time"]]))
df = df.merge(timings, on="name", how="left")
df.sort_values("time", inplace=True)
print(df)

df.to_csv("merged_results.csv", index=False)
