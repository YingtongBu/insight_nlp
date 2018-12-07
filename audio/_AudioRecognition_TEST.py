#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

import audio.AudioConverter import AudioHelper
import Common as cm
import os

if __name__ == '__main__':
  data_path = os.path.join(
    cm.get_module_path("Common"),
    "audio/test_data",
  )

  files = cm.get_files_in_folder(data_path,
                                 file_extensions=AudioHelper.)

  SpeechRecognition.audio_file_recognize(data_path,
                                         language_selection="Zh-cn")