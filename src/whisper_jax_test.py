from whisper_jax import FlaxWhisperPipline

# instantiate pipeline
pipeline = FlaxWhisperPipline("/mnt/whisper-vdi/models/models--Systran--faster-whisper-large-v3")

# # JIT compile the forward call - slow, but we only do once
# text = pipeline("audio.mp3")

# used cached function thereafter - super fast!!
text = pipeline("/mnt/whisper-vdi/data/paraini.mp3")

print(text)

# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("automatic-speech-recognition", cache_dir = '/mnt/whisper-vdi/models/', model="/mnt/whisper-vdi/models/models--Systran--faster-whisper-large-v3")