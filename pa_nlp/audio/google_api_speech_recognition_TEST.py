#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

from pa_nlp.audio.google_api_speech_recognition import audio_recognition
from pa_nlp import common as common
import os
'''
Using googleApi key
Chinese please use language="Zh-cn"
English please use language="US-en"
'''
if __name__ == '__main__':
  data_path = os.path.join(
    common.get_module_path("common"),
    "audio/audio.wav"
  )
  audio_recognition(data_path, language="Zh-cn")
