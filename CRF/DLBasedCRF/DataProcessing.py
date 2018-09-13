#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)

import random

def word_vector_generation(word_vector_file, train_jia_fang_data_file,
                           test_jia_fang_data_file):
  chinese_word_vector_file = open(word_vector_file, 'w')
  word_list = []
  for line in open(train_jia_fang_data_file):
    content = line.rstrip('\n').split('\t')[3]
    content = content.split(' ')
    for item in content:
      if item.split('/')[0] != '':
        word_list.append(item.split('/')[0])
  
  for line in open(test_jia_fang_data_file):
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

def train_model_data_set_generation(train_set_model_file, test_set_model_file,
                                    validation_set_model_file,
                                    train_jia_fang_data_file):
  file_train_obj = open(train_set_model_file, 'w')
  file_test_obj = open(test_set_model_file, 'w')
  file_validation_obj = open(validation_set_model_file, 'w')

  index = 0
  for line in open(train_jia_fang_data_file):
    content = line.rstrip('\n').split('\t')[3]
    content = content.split(' ')
    i = 0
    line_content = ''
    for item in content:
      item_list = item.split('/')
      if i == 0 and item_list[2] == 'Y':
        temp_ner = 'B-PER'
      elif item_list[2] == 'Y':
        temp_ner = 'I-PER'
      else:
        temp_ner = 'O'
      if item_list[0] != '':
        line_content += str(i + 1) + '\t' + \
                        item_list[0] + '\t' + temp_ner + '\n'
      i += 1
    if index < 3200:
      file_train_obj.write(line_content + '\n')
    elif index >= 3200 and index < 4000:
        file_test_obj.write(line_content + '\n')
    else:
      file_validation_obj.write(line_content + '\n')
    index += 1

  file_train_obj.close()
  file_test_obj.close()
  file_validation_obj.close()

def task_input_generation(predict_prob_file, id_record_file,
                          test_jia_fang_data_file, input_file):
  i = 0
  prob_list = []
  for line in open(predict_prob_file):
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

  id_record_object = open(id_record_file, 'w')
  for id_record in list(temp_dict.keys()):
    id_record_object.writelines(id_record + '\t')
  id_record_object.close()

  sentence_list = []
  index_final = 0
  for line in open(test_jia_fang_data_file):
    if str(index_final) in index_list:
      content = line.rstrip('\n').split('\t')[2]
      content = content.split(' ')
      word_list = []
      for item in content:
        if item.split('/')[0] != '':
          word_list.append(item.split('/')[0])
      sentence_list.append(word_list)
    index_final += 1
  
  input_file_object = open(input_file, 'w')
  for sentence in sentence_list:
    temp_sentence = ''
    for word in sentence:
      temp_sentence += (word + ' ')
    input_file_object.writelines(temp_sentence)
    input_file_object.write('\n')
  
  input_file_object.close()

def output_processing(id_record_file, output_file, ground_truth_file):
  id_record_object = open(id_record_file, 'r')
  id_record_list = [id_record for id_record in
                    id_record_object.read().split('\t') if id_record != '']
  
  output_object = open(output_file, 'r')
  output = [out for out in output_object.read().split('\n\n')
            if out != '']
  
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

  id_testset_list = []
  result_testset_list = []
  for line in open(ground_truth_file):
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
  print(f'accuracy: {accuracy}')

def preprocess(word_vector_file, train_jia_fang_data_file,
               test_jia_fang_data_file, train_set_model_file,
               test_set_model_file, validation_set_model_file,
               predict_prob_file, id_record_file, input_file):
  word_vector_generation(word_vector_file, train_jia_fang_data_file,
                         test_jia_fang_data_file)
  train_model_data_set_generation(train_set_model_file, test_set_model_file,
                                  validation_set_model_file,
                                  train_jia_fang_data_file)
  task_input_generation(predict_prob_file, id_record_file,
                        test_jia_fang_data_file, input_file)

def postprocess(id_record_file, output_file, ground_truth_file):
  output_processing(id_record_file, output_file, ground_truth_file)