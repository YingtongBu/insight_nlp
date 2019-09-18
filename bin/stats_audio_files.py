#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

import optparse
from pa_nlp import common as nlp
import os
import time
from pa_nlp.audio.audio_helper import AudioHelper
import multiprocessing as mp
from collections import defaultdict

def convert(audio_file: str):
  file_info = AudioHelper.get_basic_audio_info(audio_file)
  return file_info

def main():
  parser = optparse.OptionParser(usage="cmd [optons] [folder]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  parser.add_option("--file_ext", default="wav", help="default 'wav'")
  (options, args) = parser.parse_args()

  if args != []:
    audio_folder = args[0]
  else:
    audio_folder = "."

  start_time = time.time()
  files = nlp.get_files_in_folder(audio_folder, [options.file_ext], True)
  file_info = mp.Pool().map(convert, files)

  stat_channels = defaultdict(list)
  stat_sample_width = defaultdict(list)
  stat_sample_rate = defaultdict(list)
  for info in file_info:
    file_name = os.path.basename(info["file"])
    channel = info["channels"]
    sample_width = info["sample_width"]
    sample_rate = info["sample_rate"]
    stat_channels[channel].append(file_name)
    stat_sample_width[sample_width].append(file_name)
    stat_sample_rate[sample_rate].append(file_name)

  info_file = os.path.join(audio_folder, f"{options.file_ext}.info")
  with open(info_file, "w") as fou:
    print(
      f"channels: {stat_channels.keys()} "
      f"{stat_channels.values()}",
      file=fou
    )

    print(
      f"sample_width: {stat_sample_width.keys()} "
      f"{stat_sample_width.values()}",
      file=fou
    )

    print(
      f"sample rate: {stat_sample_rate.keys()} "
      f"{stat_sample_rate.values()}",
      file=fou
    )

  print(f"Time: {time.time() - start_time:.2f} sec.")

if __name__ == "__main__":
  main()
