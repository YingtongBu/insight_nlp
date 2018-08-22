#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
import speech_recognition as sr 

if __name__ == '__main__':
  r = sr.Recognizer()
  mic = sr.Microphone(device_index=0)
  with mic as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source, phrase_time_limit=10, timeout=5)
  try:
    print(r.recognize_google(audio, show_all=True, language="cmn-Hans-CN"))
  except:
    print("Sorry, I cannot understand your questions.")