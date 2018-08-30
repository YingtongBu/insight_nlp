#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
from __future__ import print_function
import re
import logging

def max_index_value(sentences, feature_name):
  max_item = 0
  for sentence in sentences:
    for entry in sentence[feature_name]:
      max_item = max(max_item, entry)
            
  return max_item

def word_normalize(word):
  word = word.lower()
  word = word.replace("--", "-")
  word = re.sub("\"+", '"', word)
  word = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}", 'DATE_TOKEN', word)
  word = re.sub("[0-9]{2}:[0-9]{2}:[0-9]{2}", 'TIME_TOKEN', word)
  word = re.sub("[0-9]{2}:[0-9]{2}", 'TIME_TOKEN', word)
  word = re.sub("[0-9.,]+", 'NUMBER_TOKEN', word)
  return word

def map_tokens_to_idx(sentences, word_to_idx):
  num_tokens = 0
  num_unknown_tokens = 0
  for sentence in sentences:
    for idx in range(len(sentence['raw_tokens'])):    
      token = sentence['raw_tokens'][idx]       
      word_idx = word_to_idx['UNKNOWN_TOKEN']
      num_tokens += 1
      if token in word_to_idx:
        word_idx = word_to_idx[token]
      elif token.lower() in word_to_idx:
        word_idx = word_to_idx[token.lower()]
      elif word_normalize(token) in word_to_idx:
        word_idx = word_to_idx[word_normalize(token)]
      else:
        num_unknown_tokens += 1
                       
      sentence['tokens'][idx] = word_idx
            
  if num_tokens > 0:
    logging.info("Unknown-Tokens: %.2f%%" % 
                 (num_unknown_tokens / float(num_tokens) * 100))