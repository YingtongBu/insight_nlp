#!/usr/bin/env python3
#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
import sys
sys.path.append("")
from Insight_NLP.CRF.BasicCRF.PreProcess import *
import optparse

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("-i", "--train_file", type=str,
                    help="the INPUT FILE of training data")
  parser.add_option("-m", "--model_type", type=str,
                    help="MODEL TYPE: company, contract, project or other")
  parser.add_option("-n", "--model_name", type=str,
                    help="MODEL NAME: save CRF model as MODEL NAME")
  (options, args) = parser.parse_args()

  # Read training file
  training_file = open(options.train_file)
  data = list()
  true_answer = list()
  sentences = list()
  index = list()
  for line in training_file:
    items = line.split('\t')
    index.append(items[0])
    true_answer.append(items[1])
    sentences.append(items[2])
    words = items[3].split(' ')[:-1]
    l = list()
    for w in words:
      if w == '':
        continue
      if w.startswith('//'):  # word is '/'
        l.append(['/'] + w[2:].split('/'))
      elif w.startswith('/'):  # word is ' '
        l.append([' '] + w[1:].split('/'))
      else:
        l.append(w.split('/'))
    data.append(l)
  training_file.close()

  # Training
  X = [extract_features(sample, options.model_type) for sample in data]
  y = [get_labels(sample) for sample in data]

  train(X, y, options.model_name)