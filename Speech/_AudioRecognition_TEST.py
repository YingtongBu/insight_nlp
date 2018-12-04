#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

import Speech.SpeechRecognition as SpeechRecognition
import Common as common
import os

if __name__ == '__main__':
  data_path = os.path.join(
    common.get_module_path("Common"),
    "Speech/audio.wav"
  )
  SpeechRecognition.audio_file_recognize(data_path,
                                         language_selection="Zh-cn")