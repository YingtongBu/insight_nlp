#!/usr/bin/env python
#coding: utf8

#Author: hong xu(hong.xu55@pactera.com)
#Last Modification: 08/20/2018

from google.cloud import translate
import os

os.environ[
    'GOOGLE_APPLICATION_CREDENTIALS'] = str(
    './NLP/Translation/ZSProject-94cb8e930aab.json')

def translate_sentence(text, target='Zh-cn'):
  # Imports the Google Cloud client library
  translate_client = translate.Client()
  translation = translate_client.translate(text, target_language=target)

  return(translation['translatedText'])

if __name__ == '__main__':
  translate_sentence('Beijing')