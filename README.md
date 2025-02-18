# whisper-vdi
Repo for the Whisper vdi transcription service

## Hyper-Parameters that have been tested using optuna.
beam_size = trial.suggest_int("beam_size", 1, 10)  
temperature = trial.suggest_float("temp", 0.1, 1.0)  
batch_size = trial.suggest_int("batch_size", 1, 32)  
best_of = trial.suggest_int("best_of", 1, 10)  
patience = trial.suggest_float("patience", 0.1, 1.0)  
chunk_length = trial.suggest_int("chunk_length", 1, 10)  
compression_ratio_threshold = trial.suggest_float("compression_ratio_threshold", 0.1, 1.0)  
repetition_penalty = trial.suggest_float("repetition_penalty", 0.1, 1.0)  
length_penalty = trial.suggest_float("length_penalty", 0.1, 1.0)  
no_speech_threshold = trial.suggest_float("no_speech_threshold", 0.1, 1.0)  
  
and prompting  

## Folder structure


## Benchmarking
[A folder in the CeR drive](https://uoa.sharepoint.com/:t:/r/sites/CentreforeResearchCeR-staff/Shared%20Documents/special-projects/Projects%202025/Nectar%20virtual%20transcription/Corpus) contains benchmark corpus and reference transcripts used for testing. Download and place them in `/benchmark`.
