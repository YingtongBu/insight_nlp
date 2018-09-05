#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

# pip install SpeechRecognition
import speech_recognition as sr

def audio_file_recognize(audio_file,
                         language_selection):
  r = sr.Recognizer()
  stock_audio = sr.AudioFile(audio_file)
  with stock_audio as source:
    audio = r.record(source)
    print(r.recognize_google(audio, language=language_selection))

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