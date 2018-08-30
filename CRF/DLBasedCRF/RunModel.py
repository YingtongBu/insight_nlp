#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
from __future__ import print_function
import nltk
from Insight_NLP.CRF.DLBasedCRF.util.PreProcessing \
  import (add_char_information, create_matrices, add_casing_information)
from Insight_NLP.CRF.DLBasedCRF.NeuralNets.BiLSTM import BiLSTM
import sys
import os

def run_model(input_file, output_file):

  model_path_list = os.listdir('./models')
  accuracy_list = []
  for model in model_path_list:
    if model.split('_')[1] == 'Store':
      accuracy_list.append(0)
    else:
      accuracy_list.append(float(model.split('_')[1]))
  model_file = './models/' + \
               model_path_list[accuracy_list.index(max(accuracy_list))]

  with open(input_file, 'r') as f:
    text = f.read()

  lstm_model = BiLSTM.load_model(model_file)
  text_list = [con for con in text.split('\n') if con != '']
  sentences = [{'tokens': nltk.word_tokenize(words)} for words in text_list]

  add_char_information(sentences)
  add_casing_information(sentences)
  data_matrix = create_matrices(sentences, lstm_model.mappings, True)

  tags = lstm_model.tag_sentences(data_matrix)
  output_file_object = open(output_file, 'w')
  for sentence_idx in range(len(sentences)):
    tokens = sentences[sentence_idx]['tokens']
    for token_idx in range(len(tokens)):
      token_tags = []
      for model_name in sorted(tags.keys()):
        token_tags.append(tags[model_name][sentence_idx][token_idx])
      output_file_object.writelines((tokens[token_idx] + '\t' + token_tags[0]))
      output_file_object.write('\n')
    output_file_object.write('\n')
  
  output_file_object.close()

if __name__ == '__main__':
  run_model('input.txt', 'output.txt')