#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

from Speech.GoogleApiSpeechRecognition import audio_recognition
import optparse

'''
Using googleApi key
Chinese please use language="Zh-cn"
English please use language="US-en"
'''
if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-a', '--audio_file',
                    default='company')
  parser.add_option('-l', '--language',
                    default='cmn-Hans-CN')
  (options, args) = parser.parse_args()
  audio_recognition(options.audio_file, options.language)