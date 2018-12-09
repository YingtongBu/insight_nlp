#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

import audio.speech_recognition as asr
import common as cm
import os

if __name__ == '__main__':
  data_path = os.path.join(
    cm.get_module_path("Common"),
    "audio/test.wav",
  )

  asr.audio_file_recognize(data_path, language_selection="Zh-cn")