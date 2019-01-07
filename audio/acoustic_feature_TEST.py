#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import common as nlp
import optparse
import os
from audio.audio_helper import AudioHelper
from audio.acoustic_feature import calc_mfcc_delta
import librosa
import tensorflow as tf

if __name__ == "__main__":
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  (options, args) = parser.parse_args()

  audio_file = os.path.join(
    nlp.get_module_path("common"),
    "audio/test_data/102-129232-0009.flac"
  )

  features = calc_mfcc_delta(audio_file, 100)
  print(f"#frame: {len(features)}, #feat: {len(features[0])}")

