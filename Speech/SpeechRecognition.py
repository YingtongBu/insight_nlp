#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

# pip install SpeechRecognition
import speech_recognition as sr 
import optparse

def listen_and_recognize(input_device_index, time_limit, 
                         time_out, language_selection):
  r = sr.Recognizer()
  mic = sr.Microphone(device_index=input_device_index)
  with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source, phrase_time_limit=time_limit, timeout=time_out)
  try:
    print(r.recognize_google(audio, show_all=True, language=language_selection))
  except:
    print("Sorry, I cannot understand your questions.")

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option('-d', '--device_index', 
                    default=1)
  parser.add_option('-p', '--phrase_time_limit', 
                    default=10)
  parser.add_option('-t', '--timeout', 
                    default=5)
  parser.add_option('-l', '--language', 
                    default='cmn-Hans-CN')                 
  (options, args) = parser.parse_args()
  listen_and_recognize(options.device_index, options.phrase_time_limit, 
                       options.timeout, options.language)