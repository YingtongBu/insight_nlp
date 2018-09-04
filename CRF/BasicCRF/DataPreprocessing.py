#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

class DataPreprocessing(object):
  def __init__(self, doc_path):
    self.doc_path = doc_path

  #code review: process_train_data():
  def train_doc_process(self):
    doc_file = open(self.doc_path)
    data = list()
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
      data.append(l)
    doc_file.close()
    print(data)
    return data

  def test_doc_process(self):
    doc_file = open(self.doc_path)
    data = list()
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
      data.append(l)
    doc_file.close()

    return data

  def get_labels(self, data):
    return [label for (token, postag, label) in data]

  #code review: get_longest_label(...)
  def get_longest(self, X, Y):
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