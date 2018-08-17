# -*- coding: utf-8 -*-
import random

# generate initial word embedding dictionary

chinese_word_vector_file = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/chinese_word_vector', 'w')

word_list = []
for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/trainTemp1.data'):
    content = line.rstrip('\n').split('\t')[3]
    content = content.split(' ')
    for item in content:
        if item.split('/')[0] != '':
            word_list.append(item.split('/')[0])

for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/testTemp1.temp'):
    content = line.rstrip('\n').split('\t')[2]
    content = content.split(' ')
    for item in content:
        if item.split('/')[0] != '':
            word_list.append(item.split('/')[0])

word_dict = dict()
for item in set(word_list):
    random_list = []
    for i in range(300):
        random_list.append(round(random.uniform(-1, 1), 6))
    word_dict[item] = random_list

for i in range(len(word_dict)):
    word_list = list(word_dict.keys())
    random_num_list = list(word_dict.values())

    vector = ''
    for index in range(len(random_num_list[i])):
        vector += (str(random_num_list[i][index]) + ' ')

    word_vector = word_list[i] + ' ' + vector
    chinese_word_vector_file.writelines(word_vector)
    chinese_word_vector_file.write('\n')

chinese_word_vector_file.close()