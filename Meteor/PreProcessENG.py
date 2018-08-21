#!/usr/bin/env python
#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from tqdm import tqdm as tq
import numpy as np

def preProcessENG(filePath):
  trainData = []
  trainX, trainY = [], []
  with open(filePath, 'r', encoding = 'latin') as newFile:
    data = newFile.readlines()[1: ]
    print(type(data))
    print('Loading data ...')
    for row in tq(data):
      row = row.split('\t')
      trainData.append([row[0], row[1]])
  for index, item in enumerate(trainData):
    if item[0].isdigit() != True:
      pass
  # check the trainY, should be from 1 - 43
