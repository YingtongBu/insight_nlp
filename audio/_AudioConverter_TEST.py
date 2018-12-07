#!/usr/bin/env python
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from audio.AudioConverter import AudioHelper
import Common as cm
import os

if __name__ == '__main__':
  data_path = os.path.join(
    cm.get_module_path("Common"),
    "audio/test_data"
  )

  files = cm.get_files_in_folder(data_path,
                                 file_extensions=AudioHelper.AUDIO_EXTENSIONS)

  for in_file in files:
    out_file = AudioHelper.convert_to_flac(in_file)
    if out_file is not None:
      length = AudioHelper.get_music_length(out_file)
      length_str = AudioHelper.seconds_to_str(length)
      print(f"[OK] {in_file} to {out_file}, {length_str}")

    else:
      print(f"[ERR] {in_file}")

