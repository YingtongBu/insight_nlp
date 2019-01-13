import librosa
import numpy as np
import common as nlp
import typing
import time
import multiprocessing as mp

# output length = (seconds) * (sample rate) / (hop_length)
HOP_LENGTH = 512

def calc_mfcc_delta(audio_file: str, mfcc_dim: int):
  '''
  :return: [mfcc, mfcc_delta, mfcc_delta2]
  '''
  wav_data, sample_rate = librosa.load(audio_file)
  mfcc = np.transpose(
    librosa.feature.mfcc(wav_data, sample_rate, n_mfcc=mfcc_dim,
                         hop_length=HOP_LENGTH),
    [1, 0]
  )

  mfcc_delta = librosa.feature.delta(mfcc)
  mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

  feature = []
  for v1, v2, v3 in zip(mfcc.tolist(), mfcc_delta.tolist(), mfcc_delta2.tolist()):
    feature.append(v1 + v2 + v3)

  return feature

def parallel_calc_features(audio_files: list, mfcc_dim: int,
                           output_folder: str, process_num: int,
                           queue_capacity: int=1024):
  def run_process(process_id: int, audio_file_pipe: mp.Queue):
    count = 0
    with open(f"{output_folder}/part.{process_id}.feat", "w") as fou:
      while True:
        audio_file = audio_file_pipe.get()
        if audio_file is None:
          print(f"run_process[{process_id}] exits!")
          break

        feature = calc_mfcc_delta(audio_file, mfcc_dim)
        print([audio_file, feature], file=fou)

        count += 1
        if count % 100 == 0:
          nlp.print_flush(f"So far, process[{process_id}] have processed "
                          f"{count} audio files.")

  assert 1 <= process_num
  nlp.execute_cmd(f"rm -r {output_folder}")
  nlp.ensure_folder_exists(output_folder)

  start_time = time.time()
  file_pipe = mp.Queue(queue_capacity)
  process_runners = [mp.Process(target=run_process, args=(idx, file_pipe))
                     for idx in range(process_num)]

  for p in process_runners:
    p.start()

  for audio_file in audio_files:
    file_pipe.put(audio_file)

  for _ in process_runners:
    file_pipe.put(None)

  for p in process_runners:
    p.join()

  duration = nlp.to_readable_time(time.time() - start_time)
  print(f"It takes {duration} to process {len(audio_files)} audio files")
