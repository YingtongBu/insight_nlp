#!/usr/bin/env python
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from audio.audio_helper import AudioHelper
import common as nlp
import os

if __name__ == '__main__':
  data_path = os.path.join(
    nlp.get_module_path("common"),
    "audio/test_data"
  )

  files = nlp.get_files_in_folder(data_path,
                                  file_extensions=AudioHelper.AUDIO_EXTENSIONS)

  for in_file in files:
    out_file = AudioHelper.convert_to_flac(in_file)
    if out_file is not None:
      length = AudioHelper.get_music_length(out_file)
      length_str = nlp.to_readable_time(length)
      print(f"[OK] {in_file} to {out_file}, {length_str}")

    else:
      print(f"[ERR] {in_file}")
      assert False

