# -*- coding: utf-8 -*-
 
# process output result and compute the accuracy
id_record_object = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/id_record.txt', 'r')
id_record_list = [id_record for id_record in id_record_object.read().split('\t') if id_record != '']

output_object = open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/output.txt', 'r')
output = [out for out in output_object.read().split('\n\n') if out != '']


output_list = []
for out in output:
    temp_out_list = out.split('\n')
    temp_output_list = []
    for temp_out in temp_out_list:
        if '-PER' in temp_out:
            temp_output_list.append(temp_out)
    output_list.append(temp_output_list)

final_output_list = []  
for output_item in output_list:
    temp_str = ''
    for item in output_item:
        temp_str += item.split('\t')[0]
    final_output_list.append(temp_str)

result_dict = dict(zip(id_record_list, final_output_list))

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

id_testset_list = []
result_testset_list = []
for line in open('/Users/xinjin/WorkSpace/workspace_vscode/LSTM_CNN_CRF_Code/LSTM_CNN_CRF/data_for_test_LSTM_CRF/label.test_lower.csv'):
    content = line.rstrip('\n').split('\t')
    id_testset_list.append(content[0].split('=')[1])
    result_testset_list.append(content[1].split('=')[1])

predict_result_list = []
for i in range(len(id_testset_list)):
    try:
        predict_result_list.append(result_dict[id_testset_list[i]])
    except:
        predict_result_list.append('')

accurate_num = 0
for i in range(len(result_testset_list)):
    if predict_result_list[i] == result_testset_list[i]:
        accurate_num += 1

accuracy = accurate_num / 1120
print(accuracy)