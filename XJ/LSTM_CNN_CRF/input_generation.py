#-*- coding: utf-8 -*-

# prepare test dataset as input

i = 0
prob_list = []
for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/predictProb1.temp'):
    prob_list.append(str(i) + '\t' + line.strip('\n'))
    i += 1

sentence_dict = dict()
for id_prob_pair in prob_list:
    temp_list = id_prob_pair.split('\t')
    if temp_list[1] not in sentence_dict.keys():
        sentence_dict[temp_list[1]] = temp_list[2]
    elif float(sentence_dict[temp_list[1]]) <= float(temp_list[2]):
        sentence_dict[temp_list[1]] = temp_list[2]

index_id_pair_dict = dict()
for id_prob_pair in prob_list:
    temp_list = id_prob_pair.split('\t')
    for id in list(sentence_dict.keys()):
        if temp_list[1] == id and temp_list[2] == sentence_dict[id]:
            index_id_pair_dict[temp_list[0]] = temp_list[1]

temp_dict = {index_id_pair_dict[key]: key for key in index_id_pair_dict}
index_list = list(temp_dict.values())

id_record_object = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/id_record.txt', 'w')
for id_record in list(temp_dict.keys()):
    id_record_object.writelines(id_record + '\t')
id_record_object.close()

sentence_list = []
index_final = 0
for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/甲方/testTemp1.temp'):
    if str(index_final) in index_list:
        content = line.rstrip('\n').split('\t')[2]
        content = content.split(' ')
        word_list = []
        for item in content:
            if item.split('/')[0] != '':
                word_list.append(item.split('/')[0])
        sentence_list.append(word_list)
    index_final += 1

input_file_object = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/input.txt', 'w')
for sentence in sentence_list:
    temp_sentence = ''
    for word in sentence:
        temp_sentence += (word + ' ')
    input_file_object.writelines(temp_sentence)
    input_file_object.write('\n')

input_file_object.close()