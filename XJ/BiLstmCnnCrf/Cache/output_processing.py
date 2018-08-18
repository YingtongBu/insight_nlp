# -*- coding: utf-8 -*-
 
# process output result and compute the accuracy
idRecordObject = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/id_record.txt', 'r')
idRecordList = [idRecord for idRecord in idRecordObject.read().split('\t') if idRecord != '']

outputObject = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/output.txt', 'r')
output = [out for out in outputObject.read().split('\n\n') if out != '']


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

# id_result_list = []
# result_pure_list = []
# for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/pureResult1.temp.csv', 'r'):
#     content = line.rstrip('\n').split(',')
#     id_result_list.append(content[0])
#     result_pure_list.append(content[1])
# print(len(id_result_list))
# print(len(result_pure_list))
# result_dict = dict(zip(id_result_list, result_pure_list))
# print(result_dict)

idTestsetList = []
resultTestsetList = []
for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/label.test_lower.csv'):
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
print(accuracy)