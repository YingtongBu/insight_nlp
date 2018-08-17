#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Time: 07/06/2018 14:37 PDT
@Author: Xin Jin
'''

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
    label_list.append([item[0], item[1]])

result_list = []
for label in label_list:
  if label[1] != '':
    result_list.append(label[0] + '\t' + label[1])

fileObj = open('/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/labelFile.txt', 'w')
for item in set(result_list):
  fileObj.writelines(item)
  fileObj.write('\n')
fileObj.close()