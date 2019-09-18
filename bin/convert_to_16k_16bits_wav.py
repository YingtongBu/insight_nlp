#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp import common as nlp
from pa_nlp.audio.audio_helper import AudioHelper
import multiprocessing as mp
import optparse
import time

def convert(audio_file: str):
  new_audio = AudioHelper.convert_to_standard_wav(audio_file)
  if new_audio is None:
    print(f"WARN: error in {audio_file}")

def main():
  parser = optparse.OptionParser(usage="cmd [optons] [audio1, ...]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  parser.add_option("--audio_folder")
  parser.add_option("--audio_type", default="wav,flac,sph",
                    help="default 'wav,flac,sph'")
  (options, args) = parser.parse_args()

  start_time = time.time()
  for audio_file in args:
    convert(audio_file)

  if options.audio_folder is not None:
    files = nlp.get_files_in_folder(
      options.audio_folder,
      options.audio_type.split(","),
      True
    )
    mp.Pool().map(convert, files)

  print(f"Time: {time.time() - start_time:.2f} sec.")

if __name__ == "__main__":
  main()
