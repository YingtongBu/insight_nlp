import librosa
import numpy as np
import common as nlp
import typing
import time
import multiprocessing as mp

def calc_mfcc_delta(audio_file: str, mfcc_dim: int):
  '''
  :return: [mfcc, mfcc_delta, mfcc_delta2]
  '''
  wav_data, sample_rate = librosa.load(audio_file)
  mfcc = np.transpose(
    librosa.feature.mfcc(wav_data, sample_rate, n_mfcc=mfcc_dim),
    [1, 0]
  )

  mfcc_delta = librosa.feature.delta(mfcc)
  mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

  feature = []
  for v1, v2, v3 in zip(mfcc.tolist(), mfcc_delta.tolist(), mfcc_delta2.tolist()):
    feature.append(v1 + v2 + v3)

  return feature

def parallel_calc_features(audio_files: list, mfcc_dim: int,
                           output_file: str, process_num: int,
                           queue_capacity: int=1024):
  def write_process(feat_data_pipe: mp.Queue):
    count = 0
    with open(output_file, "w") as fou:
      while True:
        record = feat_data_pipe.get()
        if record is None:
          break

        if 0 < count and count % 100 == 0:
          nlp.print_flush(f"So far, {count} audio files have been processed")
          fou.flush()

        name, feat = record
        print([name, feat], file=fou)
        count += 1

  def run_process(process_id: int, audio_file_pipe: mp.Queue,
                  feat_data_pipe: mp.Queue):
    while True:
      audio_file = audio_file_pipe.get()
      if audio_file is None:
        print(f"run_process[{process_id}] exits!")
        break

      feature = calc_mfcc_delta(audio_file, mfcc_dim)
      feat_data_pipe.put([audio_file, feature])

  assert 1 <= process_num

  start_time = time.time()
  audio_file_pipe = mp.Queue(queue_capacity)
  feat_data_pipe = mp.Queue(queue_capacity)
  process_runners = [mp.Process(target=run_process,
                                args=(idx, audio_file_pipe, feat_data_pipe))
                     for idx in range(process_num)]
  process_writer = mp.Process(target=write_process, args=(feat_data_pipe,))
  process_writer.start()

  for p in process_runners:
    p.start()

  for audio_file in audio_files:
    audio_file_pipe.put(audio_file)

  for _ in process_runners:
    audio_file_pipe.put(None)

  for p in process_runners:
    p.join()

  feat_data_pipe.put(None)
  process_writer.join()

  duration = nlp.to_readable_time(time.time() - start_time)
  print(f"It takes {duration} to process {len(audio_files)} audio files")
