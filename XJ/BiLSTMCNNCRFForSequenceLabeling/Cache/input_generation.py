#-*- coding: utf-8 -*-

# prepare test dataset as input

i = 0
probList = []
for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/predictProb1.temp'):
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

idRecordObject = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/id_record.txt', 'w')
for idRecord in list(tempDict.keys()):
    idRecordObject.writelines(idRecord + '\t')
idRecordObject.close()

sentenceList = []
indexFinal = 0
for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/testTemp1.temp'):
    if str(indexFinal) in indexList:
        content = line.rstrip('\n').split('\t')[2]
        content = content.split(' ')
        wordList = []
        for item in content:
            if item.split('/')[0] != '':
                wordList.append(item.split('/')[0])
        sentenceList.append(wordList)
    indexFinal += 1

inputFileObject = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/input.txt', 'w')
for sentence in sentenceList:
    tempSentence = ''
    for word in sentence:
        tempSentence += (word + ' ')
    inputFileObject.writelines(tempSentence)
    inputFileObject.write('\n')

inputFileObject.close()