#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from pa_nlp import common as nlp
import os
from pa_nlp.audio.acoustic_feature_tf import DataGraphMFCC
from pa_nlp.audio.audio_helper import AudioHelper
from pa_nlp.audio.acoustic_feature import calc_mfcc_delta
import time


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
  data_graph_mfcc = DataGraphMFCC(16000, 40)
  audio_file = os.path.join(
    nlp.get_module_path("pa_nlp.common"),
    "pa_nlp/audio/test_data/AaronHuey_2010X.sph",
  )
  wav_file = AudioHelper.convert_to_wav(audio_file)
  AudioHelper.get_basic_audio_info(AudioHelper.convert_to_flac(wav_file))

  process(data_graph_mfcc, wav_file, 40)

if __name__ == "__main__":
  main()
