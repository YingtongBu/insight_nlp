#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

import optparse
import os
from pa_nlp.audio.audio_helper import AudioHelper

def main():
  parser = optparse.OptionParser(usage="cmd [optons] [audio1, ...]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  (options, args) = parser.parse_args()
  
  for audio in args:
    print(os.path.basename(audio))
    print(AudioHelper.get_detailed_audio_info(audio))
    print()
    print(AudioHelper.get_basic_audio_info(audio))
    print()

if __name__ == "__main__":
  main()
