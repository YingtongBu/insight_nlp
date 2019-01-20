import common as nlp
import optparse
import os
import sys
import wave
from audio.audio_helper import AudioHelper
import collections
import tensorflow as tf
from audio.acoustic_feature import calc_mfcc_delta
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import json
import random
import time
import numpy
import numpy
import librosa

class DataGraphMFCC:
  def __init__(
    self,
    dct_coef_count: int=13,
    window_size: int=480,
    stride: int=160,
  ):
   '''
   By default settings, the returned dimension would be [100, 13], [100, 13],
   [100, 13]
   '''
   self._graph = tf.Graph()
   with self._graph.as_default():
     self._wav_file_ts = tf.placeholder(tf.string, [], name='wav_filename')
     self._frame_num = tf.placeholder(tf.int32, [])
     wav_loader = io_ops.read_file(self._wav_file_ts)
     wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
     audio_clamp = tf.clip_by_value(wav_decoder.audio, -1.0, 1.0)
     spectrogram = contrib_audio.audio_spectrogram(
       audio_clamp,
       window_size=window_size,
       stride=stride,
       magnitude_squared=True)

     feat_ts = contrib_audio.mfcc(
       spectrogram=spectrogram,
       sample_rate=wav_decoder.sample_rate,
       dct_coefficient_count=dct_coef_count,
     )
     self._feat_ts = feat_ts[0]
     shape = tf.shape(self._feat_ts)

     self._expanded_feat_ts = tf.pad(
       self._feat_ts,
       [[0, self._frame_num - shape[0]], [0, 0]],
     )

   self._sess = tf.Session(graph=self._graph)

  def run(self, wav_file: str, target_frame_num: int=-1):
    if target_frame_num <= 0:
      mfcc = self._sess.run(
        fetches=self._feat_ts,
        feed_dict={
          self._wav_file_ts: wav_file
        }
      )
    else:
      mfcc = self._sess.run(
        fetches=self._expanded_feat_ts,
        feed_dict={
          self._wav_file_ts: wav_file,
          self._frame_num: target_frame_num
        }
      )

    mfcc_delta1 = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc_delta1)

    return [mfcc, mfcc_delta1, mfcc_delta2]