#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp import common as nlp
import os
from pa_nlp.audio.acoustic_feature_tf import DataGraphMFCC
from pa_nlp.audio.audio_helper import AudioHelper
from pa_nlp.audio.acoustic_feature import calc_mfcc_delta
import time

def main():
  data_graph_mfcc = DataGraphMFCC(16000, 40)
  audio_file = os.path.join(
    nlp.get_module_path("pa_nlp.common"),
    "pa_nlp/audio/test_data/AaronHuey_2010X.sph",
  )
  wav_file = AudioHelper.convert_to_wav(audio_file)
  wav_file_16bits = AudioHelper.convert_to_16bits(wav_file)
  print(AudioHelper.get_basic_audio_info(wav_file_16bits))

  audio = data_graph_mfcc.read_16bits_wav_file(wav_file_16bits)
  feats = data_graph_mfcc.calc_feats(audio)
  print(feats[3])

if __name__ == "__main__":
  main()
