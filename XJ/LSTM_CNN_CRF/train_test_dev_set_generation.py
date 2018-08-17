# -*- coding: utf-8 -*-

# generate train, dev and test dataset

file_train_obj = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data/Chinese/train.txt', 'w')
file_test_obj = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data/Chinese/test.txt', 'w')
file_dev_obj = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data/Chinese/dev.txt', 'w')

index = 0
for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/trainTemp1.data'):
    content = line.rstrip('\n').split('\t')[3]
    content = content.split(' ')
    i = 0
    line_content = ''
    for item in content:
        item_list = item.split('/')
        if i == 0 and item_list[2] == 'Y':
            temp_NER = 'B-PER'
        elif item_list[2] == 'Y':
            temp_NER = 'I-PER'
        else:
            temp_NER = 'O'
        if item_list[0] != '':
            line_content += str(i + 1) + '\t' + item_list[0] + '\t' + temp_NER + '\n'
        i += 1
    if index < 3200:
        file_train_obj.write(line_content + '\n')
    elif index >= 3200 and index < 4000:
        file_test_obj.write(line_content + '\n')
    else:
        file_dev_obj.write(line_content + '\n')
    index += 1

file_train_obj.close()
file_test_obj.close()
file_dev_obj.close()