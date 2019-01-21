#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import common as nlp
import optparse
import os
import sys
import wave
from audio.acoustic_feature_tf import DataGraphMFCC
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

def process(data_graph_mfcc: DataGraphMFCC, wav_file: str, mfcc: int):
  flac_file = AudioHelper.convert_to_flac(wav_file)
  file_info = AudioHelper.get_basic_audio_info(flac_file)
  file_length = file_info["duration"]
  print(f"#length: {file_length}")

  start_time = time.time()
  feat_mfcc, feat_delta1, feat_delta2, real_len = data_graph_mfcc.run(wav_file)
  print(f"tensorflow: {time.time() - start_time}")
  print(len(feat_mfcc), len(feat_mfcc[0]))
  print(f"#frame/sec: {len(feat_mfcc) / file_length}")

  start_time = time.time()
  feat = calc_mfcc_delta(flac_file, mfcc)
  print(len(feat), len(feat[0]))
  print(f"python: {time.time() - start_time}")

def main():
  data_graph_mfcc = DataGraphMFCC(dct_coef_count=40)
  wav_file = os.path.join(
    nlp.get_module_path("common"),
    "audio/test_data/AaronHuey_2010X.sph.wav",
  )
  AudioHelper.get_basic_audio_info(AudioHelper.convert_to_flac(wav_file))

  process(data_graph_mfcc, wav_file, 40)

if __name__ == "__main__":
  main()
