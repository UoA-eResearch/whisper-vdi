from faster_whisper import WhisperModel, BatchedInferencePipeline
import time
import os
import torch
from torchmetrics.text import CharErrorRate
import optuna
import pandas as pd
import numpy as np
import jiwer

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

def whisper_run(model_name, batch_size, beam_size, name=""):
    model_dir = "/mnt/whisper-vdi/models/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = '/mnt/whisper-vdi/data/'
    audio_file = path + "cerebral.mp3"
    num_threads = os.cpu_count()

    audio_fname = audio_file.removeprefix(path)
    audio_fname = audio_fname.split('.')[0]

    print(f"*************** [INFO] NUM THREADS: {num_threads} ******************")
    print(f"*************** [INFO] DEVICE: {device} ******************")
    print(f"*************** [INFO] BATCH SIZE: {batch_size} ******************")
    print(f"*************** [INFO] BEAM SIZE: {beam_size} ******************")

    #model_names = ["large-v3-turbo", "large-v3", "large-v2"]

    #for model_name in model_names:
    print(f"*************** [INFO] MODEL: {model_name} ******************")
    start = time.time()
    compute_type = "float32"
    model = WhisperModel(model_name, 
                        download_root=model_dir,
                        compute_type=compute_type,
                        cpu_threads=num_threads)

    # segments, info = model.transcribe(audio_file)
    batched_model = BatchedInferencePipeline(model=model)

    segments, info = batched_model.transcribe(audio_file,
                                                batch_size=batch_size,
                                                beam_size=beam_size,
                                                chunk_length=1)

    with open(path + model_name + '.txt', 'w') as f:
        #f.write(transcript)
        for segment in segments:
            f.write(segment.text)
        f.close()
        #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    end = time.time()
    runtime = end - start
    print(runtime)

    gt = open(path + audio_fname + ".txt", "rt").read()
    gt = str(gt).lower()
    preds = open(path + model_name + '.txt', "rt").read()
    preds = str(preds).lower()

    metrics = get_metrics(gt, preds, model_name)

    df_data=[model_name, device, num_threads, runtime, batch_size, 
                beam_size, metrics["wer"], metrics["mer"], metrics["wil"], metrics["cer"], metrics["untransformed_cer"]]
    df_data = np.reshape(np.array(df_data), (1, -1))
    df = pd.DataFrame(data=df_data)

    # with open(path + 'optuna_preds_' + audio_fname + '.csv', 'a') as f:
    #     txt = model_name + ',Device= ' + str(device) + ',Batch_size= ' + str(batch_size) + ',num_cores= ' + str(num_threads) + ',CeR= ' + str(res.numpy()) + ',runtime= ' + str(runtime)
    #     f.write(txt + '\n')
    # f.close()

    df.to_csv(path + name + 'preds_' + audio_fname + '.csv', mode='a', header=False, index=False)
    return metrics["untransformed_cer"]

def objective(trial):
    beam_size = trial.suggest_int("beam_size", 1, 2)
    #temp = trial.suggest_float("temp", 0.1, 0.2)
    batch_size = trial.suggest_int("batch_size", 1, 2)
    #best_of = trial.suggest_int("best_of", 1, 10)
    #patience = trial.suggest_float("patience", 0.5, 1.0)
    #chunk_length = trial.suggest_int("chunk_length", 1, 10)
    #comp_ratio_threshold = trial.suggest_float("comp_ratio_threshold", 0.1, 1.0)
    #rep_penalty = trial.suggest_float("rep_penalty", 0.1, 1.0)
    #len_penalty = trial.suggest_float("len_penalty", 0.1, 1.0)
    #no_speech_thres = trial.suggest_float("no_speech_thres", 0.1, 1.0)
    model_name = "large-v3-turbo"

    untransformed_cer = whisper_run(model_name,
                                    batch_size,
                                    beam_size)#,
                                    #best_of,
                                    #chunk_length)

    return untransformed_cer


if __name__ == "__main__":
    model_dir = "/mnt/whisper-vdi/models/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = '/mnt/whisper-vdi/data/'
    audio_file = path + "cerebral.mp3"
    num_threads = os.cpu_count()

    audio_fname = audio_file.removeprefix(path)
    audio_fname = audio_fname.split('.')[0]

    ref = open(path + audio_fname + ".txt", "rt").read()
    hyp = open(path + audio_fname + "-canary.txt", "rt").read()

    met = get_metrics(ref, hyp)

    whisper_run('large-v3', 1, 1)

    # if not os.path.exists(path + 'optuna_preds_simplified_' + audio_fname + '.csv'):

    #     heads = ['Model', 'Device', 'num_threads', 'runtime', 'batch_size', 
    #             'beam_size', "wer", "mer", "wil", "cer", "untransformed_cer"]
    #     heads = np.reshape(np.array(heads), (1, -1))
    #     df = pd.DataFrame(data=heads)
    #     df.to_csv(path + 'optuna_preds_' + audio_fname + '.csv', mode='a', header=False, index=False)
   
    # study = optuna.create_study('sqlite:////mnt/whisper-vdi/data/optuna_HPO_db.db', direction="minimize")
    # study.optimize(objective, n_trials=10, n_jobs=5)
    # print(study.best_params)
   
