# -*- coding: utf-8 -*-

# generate train, dev and test dataset

fileTrainObj = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data/Chinese/train.txt', 'w')
fileTestObj = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data/Chinese/test.txt', 'w')
fileDevObj = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data/Chinese/dev.txt', 'w')

index = 0
for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/trainTemp1.data'):
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
        fileDevObj.write(lineContent + '\n')
    index += 1

fileTrainObj.close()
fileTestObj.close()
fileDevObj.close()