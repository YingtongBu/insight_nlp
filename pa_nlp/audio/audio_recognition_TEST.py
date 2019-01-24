#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

import pa_nlp.audio.speech_recognition as asr
from pa_nlp import common as cm
import os

if __name__ == '__main__':
  data_path = os.path.join(
    cm.get_module_path("common"),
    "audio/test.wav",
  )

  asr.audio_file_recognize(data_path, language_selection="Zh-cn")
