#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)

import sys
sys.path.append("")
from InforExtractionFramework.CRF.PreProcess import *
import optparse
import pycrfsuite

if __name__ == '__main__':
  usage = "usage: %prog [options]"
  parser = optparse.OptionParser(usage=usage)
  parser.add_option("-t", "--testFile", type=str,
                    help="the INPUT FILE of testing data")
  parser.add_option("-o", "--outputFile", type=str,
                    help="the OUTPUT FILE of predictions")
  parser.add_option("-m", "--modelType", type=str,
                    help="MODEL TYPE: company, contract, project or other")
  parser.add_option("-n", "--modelName", type=str,
                    help="MODEL NAME: use MODEL NAME to predict")
  (options, args) = parser.parse_args()

  # Read testing file
  testingFile = open(options.testFile)
  data = list()
  sentences = list()
  index = list()
  for line in testingFile:
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
  testingFile.close()

  # Predict
  xTest = [extractFeatures(sample, options.modelType) for sample in data]

  tagger = pycrfsuite.Tagger()
  tagger.open(options.modelName)
  yPred = list()
  probability = list()
  for xSeq in xTest:
    pred = (tagger.tag(xSeq))
    yPred.append(pred)
    probability.append(tagger.probability(pred))

  result = list()
  for i in range(len(yPred)):
    result.append(getLongest([x[1].split("=")[1]
                              for x in xTest[i]], yPred[i]))

  f = open(options.outputFile, 'w+')
  for i in range(len(result)):
    f.write(index[i] + ' ' + result[i] + ' ' + str(probability[i]) + '\n')
  f.close()
