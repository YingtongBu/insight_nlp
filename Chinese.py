#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Insight_NLP.Common import *

import jieba
import jieba.posseg as pseg

def convert_full_to_half(s):
  '''全角转半角'''
  n = []
  for char in s:
    num = ord(char)
    if num == 12288:
      num = 32
    elif num == 12290:
      num = 46
    elif 65281 <= num <= 65374:
      num -= 65248
    num = chr(num)
    n.append(num)
  return ''.join(n)

def segment_sentence(text, pos_tagging=False):
  if pos_tagging:
    words, tags = [], []
    for token in pseg.cut(text):
      words.append(token.word)
      tags.append(token.flag)
    return words, tags
  else:
    return list(jieba.cut(text, cut_all=False))
