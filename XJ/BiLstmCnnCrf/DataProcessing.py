# -*- coding: utf-8 -*-
import random

def wordVectorGeneration(wordVectorFile, trainJiaFangDataFile, 
                         testJiaFangDataFile):
  chineseWordVectorFile = open(wordVectorFile, 'w')
  wordList = []
  for line in open(trainJiaFangDataFile):
    content = line.rstrip('\n').split('\t')[3]
    content = content.split(' ')
    for item in content:
      if item.split('/')[0] != '':
        wordList.append(item.split('/')[0])
  
  for line in open(testJiaFangDataFile):
    content = line.rstrip('\n').split('\t')[2]
    content = content.split(' ')
    for item in content:
      if item.split('/')[0] != '':
        wordList.append(item.split('/')[0])
  
  wordDict = dict()
  for item in set(wordList):
    randomList = []
    for i in range(300):
      randomList.append(round(random.uniform(-1, 1), 6))
    wordDict[item] = randomList
  
  for i in range(len(wordDict)):
    wordList = list(wordDict.keys())
    randomNumList = list(wordDict.values())
    vector = ''
    for index in range(len(randomNumList[i])):
      vector += (str(randomNumList[i][index]) + ' ')
    wordVector = wordList[i] + ' ' + vector
    chineseWordVectorFile.writelines(wordVector)
    chineseWordVectorFile.write('\n')
  
  chineseWordVectorFile.close()

def trainModelDataSetGeneration(trainSetModelFile, testSetModelFile, 
                                validationSetModelFile, trainJiaFangDataFile): 
  fileTrainObj = open(trainSetModelFile, 'w')
  fileTestObj = open(testSetModelFile, 'w')
  fileValidationObj = open(validationSetModelFile, 'w')

  index = 0
  for line in open(trainJiaFangDataFile):
    content = line.rstrip('\n').split('\t')[3]
    content = content.split(' ')
    i = 0
    lineContent = ''
    for item in content:
      itemList = item.split('/')
      if i == 0 and itemList[2] == 'Y':
        tempNER = 'B-PER'
      elif itemList[2] == 'Y':
        tempNER = 'I-PER'
      else:
        tempNER = 'O'
      if itemList[0] != '':
        lineContent += str(i + 1) + '\t' + itemList[0] + '\t' + tempNER + '\n'
      i += 1
    if index < 3200:
      fileTrainObj.write(lineContent + '\n')
    elif index >= 3200 and index < 4000:
        fileTestObj.write(lineContent + '\n')
    else:
      fileValidationObj.write(lineContent + '\n')
    index += 1

  fileTrainObj.close()
  fileTestObj.close()
  fileValidationObj.close()

def taskInputGeneration(predictProbFile, idRecordFile, testJiaFangDataFile, 
                        inputFile): 
  i = 0
  probList = []
  for line in open(predictProbFile):
    probList.append(str(i) + '\t' + line.strip('\n'))
    i += 1
  
  sentenceDict = dict()
  for idProbPair in probList:
    tempList = idProbPair.split('\t')
    if tempList[1] not in sentenceDict.keys():
      sentenceDict[tempList[1]] = tempList[2]
    elif float(sentenceDict[tempList[1]]) <= float(tempList[2]):
      sentenceDict[tempList[1]] = tempList[2]

  indexIdPairDict = dict()
  for idProbPair in probList:
    tempList = idProbPair.split('\t')
    for id in list(sentenceDict.keys()):
      if tempList[1] == id and tempList[2] == sentenceDict[id]:
        indexIdPairDict[tempList[0]] = tempList[1]
  
  tempDict = {indexIdPairDict[key]: key for key in indexIdPairDict}
  indexList = list(tempDict.values())

  idRecordObject = open(idRecordFile, 'w')
  for idRecord in list(tempDict.keys()):
    idRecordObject.writelines(idRecord + '\t')
  idRecordObject.close()

  sentenceList = []
  indexFinal = 0
  for line in open(testJiaFangDataFile):
    if str(indexFinal) in indexList:
      content = line.rstrip('\n').split('\t')[2]
      content = content.split(' ')
      wordList = []
      for item in content:
        if item.split('/')[0] != '':
          wordList.append(item.split('/')[0])
      sentenceList.append(wordList)
    indexFinal += 1
  
  inputFileObject = open(inputFile, 'w')
  for sentence in sentenceList:
    tempSentence = ''
    for word in sentence:
      tempSentence += (word + ' ')
    inputFileObject.writelines(tempSentence)
    inputFileObject.write('\n')
  
  inputFileObject.close()

def outputProcessing(idRecordFile, outputFile, groundTruthFile):
  idRecordObject = open(idRecordFile, 'r')
  idRecordList = [idRecord for idRecord in idRecordObject.read().split('\t') 
                  if idRecord != '']
  
  outputObject = open(outputFile, 'r')
  output = [out for out in outputObject.read().split('\n\n') 
            if out != '']
  
  outputList = []
  for out in output:
    tempOutList = out.split('\n')
    tempOutputList = []
    for tempOut in tempOutList:
      if '-PER' in tempOut:
        tempOutputList.append(tempOut)
    outputList.append(tempOutputList)
  
  finalOutputList = []
  for outputItem in outputList:
    tempStr = ''
    for item in outputItem:
      tempStr += item.split('\t')[0]
    finalOutputList.append(tempStr)
  
  resultDict = dict(zip(idRecordList, finalOutputList))

  idTestsetList = []
  resultTestsetList = []
  for line in open(groundTruthFile):
    content = line.rstrip('\n').split('\t')
    idTestsetList.append(content[0].split('=')[1])
    resultTestsetList.append(content[1].split('=')[1])
  
  predictResultList = []
  for i in range(len(idTestsetList)):
    try:
      predictResultList.append(resultDict[idTestsetList[i]])
    except:
      predictResultList.append('')
  
  accurateNum = 0
  for i in range(len(resultTestsetList)):
    if predictResultList[i] == resultTestsetList[i]:
      accurateNum += 1
  
  accuracy = accurateNum / 1120
  print('accuracy = ', accuracy)

def preprocess(wordVectorFile, trainJiaFangDataFile, 
               testJiaFangDataFile, trainSetModelFile,
               testSetModelFile, validationSetModelFile,
               predictProbFile, idRecordFile, inputFile):
  wordVectorGeneration(wordVectorFile, trainJiaFangDataFile, 
                       testJiaFangDataFile)
  trainModelDataSetGeneration(trainSetModelFile, testSetModelFile, 
                              validationSetModelFile, trainJiaFangDataFile)
  taskInputGeneration(predictProbFile, idRecordFile, testJiaFangDataFile, 
                      inputFile)

def postprocess(idRecordFile, outputFile, groundTruthFile):
  outputProcessing(idRecordFile, outputFile, groundTruthFile)

if __name__ == '__main__':
  pass