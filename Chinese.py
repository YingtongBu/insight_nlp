#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Common import *

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
  
def split_and_norm_string(text: str):
  '''
  Tokenization/string cleaning for Chinese and English mixed data
  '''
  text = convert_full_to_half(text)
  
  text = re.sub(r"[^A-Za-z0-9\u4e00-\u9fa5()（）！？，,!?\'\`]", " ", text)
  text = re.sub(r"\'s", " \'s", text)
  text = re.sub(r"\'ve", " \'ve", text)
  text = re.sub(r"n\'t", " n\'t", text)
  text = re.sub(r"\'re", " \'re", text)
  text = re.sub(r"\'d", " \'d", text)
  text = re.sub(r"\'ll", " \'ll", text)
  text = re.sub(r",", " , ", text)
  text = re.sub(r"!", " ! ", text)
  text = re.sub(r"\(", " \( ", text)
  text = re.sub(r"\)", " \) ", text)
  text = re.sub(r"\?", " \? ", text)
  text = re.sub(r"\s{2,}", " ", text)
  # clean for chinese character
  new_string = ""
  for char in text:
    if re.findall(r"[\u4e00-\u9fa5]", char) != []:
      char = " " + char + " "
    new_string += char
    
  return new_string.strip().lower().replace('\ufeff', '')
