from faster_whisper import WhisperModel, BatchedInferencePipeline
import time
import os
import torch
from torchmetrics.text import CharErrorRate

start = time.time()

model_dir = "/home/ubuntu/whisper-vdi/models/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '/home/ubuntu/whisper-vdi/data/'
audio_file = path + "paraini.m4a" #"/home/ubuntu/whisper-vdi/data/foundationnorth.mp3"
batch_size = 1 # reduce if low on GPU mem
num_threads = os.cpu_count()

print(f"*************** [INFO] NUM THREADS: {num_threads} ******************")

model_names = ["large-v3-turbo", "large-v3"]

for model_name in model_names:
    compute_type = "float32"
    model = WhisperModel(model_name, 
                        download_root=model_dir,
                        compute_type=compute_type,
                        cpu_threads=num_threads)

    # segments, info = model.transcribe(audio_file)
    batched_model = BatchedInferencePipeline(model=model)
    segments, info = batched_model.transcribe(audio_file, batch_size=batch_size)

    with open(path + model_name + '.txt' , 'a') as f:
        #f.write(transcript)
        for segment in segments:
            f.write(segment.text)
        f.close()
        #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    end = time.time()
    runtime = end - start
    print(runtime)

    audio_fname = audio_file.removeprefix(path)
    audio_fname = audio_fname.split('.')[0]

    gt = open(path + "paraini.txt", "rt")
    gt = str(gt).lower()
    preds = open(path + model_name + '.txt', "rt")
    preds = str(preds).lower()

    cer = CharErrorRate()
    res = cer(preds, gt)
    print(str(res.numpy()))

    with open(path + 'transcription_times_' + audio_fname + '.csv', 'a') as f:
        txt = model_name + ',Device= ' + str(device) + ',Batch_size= ' + str(batch_size) + ',num_cores= ' + str(num_threads) + ',CeR= ' + str(res.numpy()) + ',runtime= ' + str(runtime)
        f.write(txt + '\n')
    f.close()


