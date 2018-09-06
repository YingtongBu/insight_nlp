#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

from Insight_NLP.Chinese import *
class DataPreprocessing(object):

  def __init__(self, data):
    self.data = data

  def process_train_data(self):
    doc_file = open(self.data)
    sample = list()
    for line in doc_file:
      words = line.split(' ')[:-1]
      l = list()
      for w in words:
        if w == '':
          continue
        if w.startswith('//'):
          l.append(['/'] + w[2:].split('/'))
        elif w.startswith('/'):
          l.append([' '] + w[1:].split('/'))
        else:
          l.append(w.split('/'))
      sample.append(l)
    doc_file.close()
    return sample

  def process_test_data(self):
    doc_file = open(self.data)
    sample = list()
    for line in doc_file:
      words = line.split(' ')[:-1]
      l = list()
      for w in words:
        if w == '':
          continue
        if w.startswith('//'):
          l.append(['/', 'x'])
        elif w.startswith('/'):
          l.append([' ', 'x'])
        else:
          l.append(w.split('/'))
      sample.append(l)
    doc_file.close()

    return sample

  def get_labels(self, sample):
    return [label for (token, postag, label) in sample]

  def get_longest_label(self, X, Y):
    n = len(Y)
    current_len = 0
    max_len = 0
    prev_index = -2
    current_start = -1
    max_start = -1

    for i in range(n):
      if Y[i] == "Y":
        if prev_index == i - 1:
          current_len += 1
          prev_index = i
        else:
          prev_index = i
          current_len = 1
          current_start = i
        if current_len > max_len:
          max_len = current_len
          max_start = current_start

    if max_start == -1:
      return ''
    else:
      return ''.join(X[max_start:(max_start + max_len)])