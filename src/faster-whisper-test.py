from faster_whisper import WhisperModel, BatchedInferencePipeline
import time
import os
import torch

start = time.time()

model_dir = "/home/ubuntu/whisper-vdi/models/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audio_file = "/home/ubuntu/whisper-vdi/data/foundationnorth.mp3"
batch_size = 32 # reduce if low on GPU mem
if device == 'cpu':
    compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
else:
    compute_type = "float16"
num_threads = os.cpu_count()

print(f"*************** [INFO] NUM THREADS: {num_threads} ******************")

compute_type = "float16"

model = WhisperModel("large-v3-turbo", 
                     download_root=model_dir, 
                     local_files_only=True,
                     compute_type=compute_type,
                     cpu_threads=num_threads)

# segments, info = model.transcribe(audio_file)
batched_model = BatchedInferencePipeline(model=model)
segments, info = batched_model.transcribe(audio_file, batch_size=batch_size)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

end = time.time()

print(end-start)