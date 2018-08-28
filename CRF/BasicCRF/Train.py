#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)

import sys
sys.path.append("")
from InforExtractionFramework.CRF.PreProcess import *
import optparse

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("-i", "--trainFile", type=str,
                    help="the INPUT FILE of training data")
  parser.add_option("-m", "--modelType", type=str,
                    help="MODEL TYPE: company, contract, project or other")
  parser.add_option("-n", "--modelName", type=str,
                    help="MODEL NAME: save CRF model as MODEL NAME")
  (options, args) = parser.parse_args()

  # Read training file
  trainingFile = open(options.trainFile)
  data = list()
  trueAnswer = list()
  sentences = list()
  index = list()
  for line in trainingFile:
    items = line.split('\t')
    index.append(items[0])
    trueAnswer.append(items[1])
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
  trainingFile.close()

  # Training
  X = [extractFeatures(sample, options.modelType) for sample in data]
  y = [getLabels(sample) for sample in data]

  train(X, y, options.modelName)