#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

import Speech.SpeechRecognition as SpeechRecognition
import optparse
if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-a', '--audio_file',
                    default='company')
  parser.add_option('-l', '--language',
                    default='cmn-Hans-CN')
  (options, args) = parser.parse_args()
  SpeechRecognition.audio_file_recognize(options.audio_file,
                                         options.language)