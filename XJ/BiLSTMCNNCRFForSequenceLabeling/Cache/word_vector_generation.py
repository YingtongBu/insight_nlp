# -*- coding: utf-8 -*-
import random

chineseWordVectorFile = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/chinese_word_vector', 'w')

wordList = []
for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/trainTemp1.data'):
    content = line.rstrip('\n').split('\t')[3]
    content = content.split(' ')
    for item in content:
        if item.split('/')[0] != '':
            wordList.append(item.split('/')[0])

for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/testTemp1.temp'):
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