#coding: utf8
#author: Shuang Zhao (shuang.zhao11@pactera.com)

import numpy as np

def openTrainData(trainFile, lowRate, num):
  trainX, trainY = [], []
  rawText = []
  for line in open(trainFile, 'r').readlines():
    if int(line.replace('\ufeff', '').replace('\n', '').split('\t')[0]) < 45:
      trainX.append((line.replace('\ufeff', '').replace('\n', '').split('\t')[1]
                     + '*' * 200)[:num])
      trainY.append(
        int(line.replace('\ufeff', '').replace('\n', '').split('\t')[0]))
      rawText.append(line.replace('\ufeff', '').replace('\n', '').split('\t'))
  wordDict = {}
  wordCount = {}
  index = 0
  for line in trainX:
    wordSeparate = []
    for idx in range(len(line)):
      wordSeparate.append(line[idx])
    wordSeparate = sorted(list(set(wordSeparate)))
    for char in wordSeparate:
      if char not in wordCount:
        wordCount[char] = 1
      else:
        wordCount[char] += 1
      if char not in wordDict and wordCount[char] > lowRate:
        wordDict[char] = index
        index += 1
  dictLength = len(wordDict)
  print('length of wordDict:', len(wordDict))
  trainOutputX = embedding(trainX, wordDict)
  del trainX
  trainOutputY = oneHot(trainY)
  del trainY
  return trainOutputX, trainOutputY, dictLength, wordDict, rawText

def embedding(data, wordDict):
  dataOutput = []
  for line in data:
    wordSeparate = []
    for idx in range(len(line)):
      wordSeparate.append(line[idx])
    embeddingWord = []
    for char in wordSeparate:
      if char in wordDict.keys():
        embeddingWord.append(wordDict[char])
      else:
        embeddingWord.append(len(wordDict))
    dataOutput.append(np.array(embeddingWord))
  dataOutput = np.array(dataOutput)
  return dataOutput

def oneHot(data):
  largest = max(data)
  out = []
  for idx in range(len(data)):
    temp = np.zeros(largest + 1)
    temp[data[idx]] += 1
    out.append(temp)
  out = np.array(out)
  return out

def batchIter(data, batchSize, numEpochs, shuffle=True):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  dataSize = len(data)
  numBatchesPerEpoch = int((len(data) - 1) / batchSize) + 1
  for epoch in range(numEpochs):
    # Shuffle the data at each epoch
    if shuffle:
      shuffleIndices = np.random.permutation(np.arange(dataSize))
      shuffledData = data[shuffleIndices]
    else:
      shuffledData = data
    for batchNum in range(numBatchesPerEpoch):
      startIndex = batchNum * batchSize
      endIndex = min((batchNum + 1) * batchSize, dataSize)
      yield shuffledData[startIndex:endIndex]

if __name__ == '__main__':
  pass