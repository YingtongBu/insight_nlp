#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
from __future__ import print_function
import nltk
from util.PreProcessing import (addCharInformation, createMatrices, 
                                addCasingInformation)
from NeuralNets.BiLSTM import BiLSTM
import sys
import os

def runModel(inputFile, outputFile):

  modelPathList = os.listdir('./Models')
  accuracyList = []
  for model in modelPathList:
    if model.split('_')[1] == 'Store':
      accuracyList.append(0)
    else:
      accuracyList.append(float(model.split('_')[1]))
  modelFile = './Models/' + modelPathList[accuracyList.index(max(accuracyList))]

  with open(inputFile, 'r') as f:
    text = f.read()

  lstmModel = BiLSTM.loadModel(modelFile)
  textList = [con for con in text.split('\n') if con != '']
  sentences = [{'tokens': nltk.word_tokenize(words)} for words in textList]

  addCharInformation(sentences)
  addCasingInformation(sentences)
  dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

  tags = lstmModel.tagSentences(dataMatrix)
  outputFileObject = open(outputFile, 'w')
  for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']
    for tokenIdx in range(len(tokens)):
      tokenTags = []
      for modelName in sorted(tags.keys()):
        tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])  
      outputFileObject.writelines((tokens[tokenIdx] + '\t' + tokenTags[0]))
      outputFileObject.write('\n')
    outputFileObject.write('\n')
  
  outputFileObject.close()

if __name__ == '__main__':
  runModel('input.txt', 'output.txt')