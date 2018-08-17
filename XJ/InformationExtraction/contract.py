# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import unicodedata
import os

read_path = '/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/round1_train_20180518/重大合同/html/'
file_list = os.listdir(read_path)
document_dict = dict()

for file in file_list:
    file_object = open(read_path + file, 'r')    
    html_context = file_object.read()
    soup = BeautifulSoup(html_context, 'html5lib')
    sentence_list = []
    for content in soup.find_all('div', attrs={"type" : "content"}):
        content = unicodedata.normalize('NFKC', content.text.replace('\n', ' ').replace(' ', ''))
        if content != '':
            content_list = content.split('。')
            for sentence in content_list:
                if sentence != '':
                    sentence_list.append(sentence)
    document_dict[file[: -5]] = sentence_list

train_path = '/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/round1_train_20180518/重大合同/hetong.train'
with open(train_path) as train_file:
    result_list = []
    for line in train_file.readlines():
        result_list.append(line.rstrip('\n'))

final_list = []
for result in result_list:
    temp_list = result.split('\t')
    final_list.append(temp_list)

label_list = []
for item in final_list:
    if len(item) > 4:
        label_list.append([item[0], item[1], item[3], item[4]])
    else:
        label_list.append([item[0], item[1], item[3], ''])
'''
Get classification training sets.
'''
company_list = []
for label in label_list:
    for sentence in document_dict[label[0]]:
        if label[1] != '' and label[1] in sentence:
            company_list.append(sentence + '\t' + '1')
        else:
            company_list.append(sentence + '\t' + '0')

file_write_obj = open('/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/train_company.txt', 'w')
for item in set(company_list):
    file_write_obj.writelines(item)
    file_write_obj.write('\n')
file_write_obj.close()

project_list = []
for label in label_list:
    for sentence in document_dict[label[0]]:
        if label[2] != '' and label[2] in sentence:
            project_list.append(sentence + '\t' + '1')
        else:
            project_list.append(sentence + '\t' + '0')

file_write_obj = open('/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/train_project.txt', 'w')
for item in set(project_list):
    file_write_obj.writelines(item)
    file_write_obj.write('\n')
file_write_obj.close()

contract_list = []
for label in label_list:
    for sentence in document_dict[label[0]]:
        if label[3] != '' and label[3] in sentence:
            contract_list.append(sentence + '\t' + '1')
        else:
            contract_list.append(sentence + '\t' + '0')

file_write_obj = open('/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/train_contract.txt', 'w')
for item in set(contract_list):
    file_write_obj.writelines(item)
    file_write_obj.write('\n')
file_write_obj.close()           
'''
Get text labeling txt files.
'''            
# company_list = []
# for label in label_list:
#     for sentence in document_dict[label[0]]:
#         if label[1] != '' and label[1] in sentence:
#             company_list.append(label[0] + '\t' + sentence + '\t' + label[1])

# project_list = []
# for label in label_list:
#     for sentence in document_dict[label[0]]:
#         if label[2] != '' and label[2] in sentence:
#             project_list.append(label[0] + '\t' + sentence + '\t' + label[2])

# contract_list = []
# for label in label_list:
#     for sentence in document_dict[label[0]]:
#         if label[3] != '' and label[3] in sentence:
#             contract_list.append(label[0] + '\t' + sentence + '\t' + label[3])

# file_write_obj = open('/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/company.txt', 'w')
# for item in set(company_list):
#     file_write_obj.writelines(item)
#     file_write_obj.write('\n')
# file_write_obj.close()

# file_write_obj = open('/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/project.txt', 'w')
# for item in set(project_list):
#     file_write_obj.writelines(item)
#     file_write_obj.write('\n')
# file_write_obj.close()

# file_write_obj = open('/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/contract.txt', 'w')
# for item in set(contract_list):
#     file_write_obj.writelines(item)
#     file_write_obj.write('\n')
# file_write_obj.close()