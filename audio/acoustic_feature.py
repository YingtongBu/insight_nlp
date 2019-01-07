import librosa
import numpy as np
import common as nlp
import typing

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

  features = []
  for v1, v2, v3 in zip(mfcc.tolist(), mfcc_delta.tolist(), mfcc_delta2.tolist()):
    features.append(v1 + v2 + v3)

  return features

