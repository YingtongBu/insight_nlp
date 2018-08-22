#!/usr/bin/env python
#coding: utf8

#Author: hong xu(hong.xu55@pactera.com)
#Last Modification: 08/20/2018

import pandas as pd
from google.cloud import translate
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './ZSProject-94cb8e930aab.json'

def run_quickstart(text):
  # Imports the Google Cloud client library
  translate_client = translate.Client()
  target = 'Zh-cn'
  translation = translate_client.translate(text, target_language=target)

  return(translation)

if __name__ == '__main__':
  #todo: code review: should use cmdline parameter.
  file = open('./TotalEvent.txt', encoding='latin')
  eventDf = pd.DataFrame.from_csv(file, index_col=None, sep='\t')
  trans = []
  for idx, text in enumerate(eventDf['Events']):
    translation = run_quickstart(text)
    print(idx)
    trans.append(translation['translatedText'])
