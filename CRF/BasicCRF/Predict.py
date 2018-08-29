#!/usr/bin/env python3
#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
import sys
sys.path.append("")
from Insight_NLP.CRF.BasicCRF.PreProcess import *
import optparse
import pycrfsuite

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("-t", "--test_file", type=str,
                    help="the INPUT FILE of testing data")
  parser.add_option("-o", "--output_file", type=str,
                    help="the OUTPUT FILE of predictions")
  parser.add_option("-m", "--model_type", type=str,
                    help="MODEL TYPE: company, contract, project or other")
  parser.add_option("-n", "--model_name", type=str,
                    help="MODEL NAME: use MODEL NAME to predict")
  (options, args) = parser.parse_args()

  # Read testing file
  testing_file = open(options.test_file)
  data = list()
  sentences = list()
  index = list()
  for line in testing_file:
    items = line.split('\t')
    index.append(items[0])
    sentences.append(items[1])
    words = items[2].strip().split(' ')
    l = list()
    for w in words:
      if w == '':
        continue
      if w.startswith('//'):  # word is '/'
        l.append(['/', 'x'])
      elif w.startswith('/'):  # word is ' '
        l.append([' ', 'x'])
      else:
        l.append(w.split('/'))
    data.append(l)
  testing_file.close()

  # Predict
  x_test = [extract_features(sample, options.model_type) for sample in data]

  tagger = pycrfsuite.Tagger()
  tagger.open(options.model_name)
  y_pred = list()
  probability = list()
  for x_seq in x_test:
    pred = (tagger.tag(x_seq))
    y_pred.append(pred)
    probability.append(tagger.probability(pred))

  result = list()
  for i in range(len(y_pred)):
    result.append(get_longest([x[1].split("=")[1]
                              for x in x_test[i]], y_pred[i]))

  f = open(options.output_file, 'w+')
  for i in range(len(result)):
    f.write(index[i] + ' ' + result[i] + ' ' + str(probability[i]) + '\n')
  f.close()
