#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
from __future__ import print_function
import re
import logging

def max_index_value(sentences, featureName):
  maxItem = 0
  for sentence in sentences:
    for entry in sentence[featureName]:
      maxItem = max(maxItem, entry)
            
  return maxItem

def word_normalize(word):
  word = word.lower()
  word = word.replace("--", "-")
  word = re.sub("\"+", '"', word)
  word = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}", 'DATE_TOKEN', word)
  word = re.sub("[0-9]{2}:[0-9]{2}:[0-9]{2}", 'TIME_TOKEN', word)
  word = re.sub("[0-9]{2}:[0-9]{2}", 'TIME_TOKEN', word)
  word = re.sub("[0-9.,]+", 'NUMBER_TOKEN', word)
  return word

def map_tokens_to_idx(sentences, word2Idx):
  numTokens = 0
  numUnknownTokens = 0
  for sentence in sentences:
    for idx in range(len(sentence['raw_tokens'])):    
      token = sentence['raw_tokens'][idx]       
      wordIdx = word2Idx['UNKNOWN_TOKEN']
      numTokens += 1
      if token in word2Idx:
        wordIdx = word2Idx[token]
      elif token.lower() in word2Idx:
        wordIdx = word2Idx[token.lower()]
      elif word_normalize(token) in word2Idx:
        wordIdx = word2Idx[word_normalize(token)]
      else:
        numUnknownTokens += 1
                       
      sentence['tokens'][idx] = wordIdx
            
  if numTokens > 0:
    logging.info("Unknown-Tokens: %.2f%%" % 
                 (numUnknownTokens / float(numTokens) * 100))