import whisperx
import gc 
import time
import os
import torch
import jiwer
import pandas as pd
import numpy as np

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

model_dir = "/mnt/whisper-vdi/models/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '/mnt/whisper-vdi/data/'
audio_file = path + "cerebral.mp3"
num_threads = os.cpu_count()

audio_fname = audio_file.removeprefix(path)
audio_fname = audio_fname.split('.')[0]

batch_size = 1 # reduce if low on GPU mem

if device.type == 'cpu':
    compute_precision = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
else:
    compute_precision = "float16"

num_threads = os.cpu_count()
print(f"*************** [INFO] NUM THREADS: {num_threads} ******************")
print(f"*************** [INFO] Device: {device} ******************")
print(f"*************** [INFO] Precision: {compute_precision} ******************")


# 1. Transcribe with original whisper (batched)
#model = whisperx.load_model("large-v3", device, compute_type=compute_type, language='mi', threads = 32)

start = time.time()

#save model to local path (optional)
model_dir = "/mnt/whisper-vdi/models/"
#model_name = 'Systran/faster-whisper-large-v3'
model_names = ['deepdml/faster-whisper-large-v3-turbo-ct2', 'Systran/faster-whisper-large-v3']

for model_name in model_names:
    model = whisperx.load_model(model_name, 
                                device.type, 
                                compute_type=compute_precision, 
                                download_root=model_dir, 
                                #language='mi', 
                                threads = num_threads,
                                local_files_only=False)

    audio = whisperx.load_audio(audio_file)
    results = model.transcribe(audio, batch_size=batch_size)
    segments = results['segments']
    model_name = model_name.replace('/','_')

    with open(path + model_name + '_' + audio_fname +'.txt', 'w') as f:
        for segment in segments:
            f.write(segment['text'])
        f.close()
        #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    end = time.time()
    runtime = end - start
    print(runtime)

    gt = open(path + audio_fname + ".txt", "rt").read()
    gt = str(gt).lower()
    preds = open(path + model_name + '_' + audio_fname +'.txt', "rt").read()
    preds = str(preds).lower()

    metrics = get_metrics(gt, preds, model_name)
    beam_size = 'Default'

    if not os.path.exists(path + 'whisperx_preds_' + audio_fname + '.csv'):
        heads = ['Model', 'Device', 'num_threads', 'runtime', 'batch_size', 
                'beam_size', "wer", "mer", "wil", "cer", "untransformed_cer"]
        heads = np.reshape(np.array(heads), (1, -1))
        df = pd.DataFrame(data=heads)
        df.to_csv(path + 'whisperx_preds_' + audio_fname + '.csv', mode='a', header=False, index=False)

    df_data=[model_name, device.type, num_threads, runtime, batch_size, 
                beam_size, metrics["wer"], metrics["mer"], metrics["wil"], metrics["cer"], metrics["untransformed_cer"]]

    df_data = np.reshape(np.array(df_data), (1, -1))
    df = pd.DataFrame(data=df_data)

    df.to_csv(path + 'whisperx_preds_' + audio_fname + '.csv', mode='a', header=False, index=False)

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# # 2. Align whisper output
# model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
# result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

#print(result["segments"]) # after alignment

# # delete model if low on GPU resources
# # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# # 3. Assign speaker labels
# #diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

# # add min/max number of speakers if known
# #diarize_segments = diarize_model(audio)
# # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

# result = whisperx.assign_word_speakers(diarize_segments, result)
# #print(diarize_segments)
# print(result["segments"]) # segments are now assigned speaker IDs