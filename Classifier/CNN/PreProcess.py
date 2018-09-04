#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import numpy as np
from sklearn import preprocessing

def load_data(data, num_of_class:int=45):
  """
  :param data: [[10, Ping An is in Palo Alto], [44, welcome to ping an lab]]
  :param num_of_class: from 0 to num_of_class
  :return:
  """
  x_text, train_y, y = [], [], []
  data = [sample.strip() for sample in data]
  for row in data:
    row = row.split('\t')
    x_text.append(row[1].replace('\ufeff', ''))
    train_y.append(row[0])
  # Split by words
  x_original = x_text
  x_text = [token_str(sent) for sent in x_text]
  # clean y
  for item in train_y:
    try:
      item = int(item)
      if item > num_of_class:
        item = 0
    except:
      item = 0
    y.append(item)

  # Generate labels
  y = [[item] for item in y]
  enc = preprocessing.OneHotEncoder()
  enc.fit(y)
  y = enc.transform(y).toarray()
  return [x_text, y, x_original]

import re

def token_str(string):
  """
  Tokenization/string cleaning for Chinese and English data
  """
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  new_string = ""
  for char in string:
    if re.findall(r"\u4e00-\u9fa5", char) != []:
      char = " " + char + " "
    new_string += char
  return new_string.strip().lower()

def load_data_chinese(data, low_rate, len_sentence, num_of_class:int=45):
  train_x, train_y = [], []
  raw_text = []
  for line in data:
    if int(line.replace('\ufeff', '').replace('\n', '').split('\t')[0]) \
                                                                < num_of_class:
      train_x.append((line.replace('\ufeff', '').replace('\n', '').split(
        '\t')[1] + '*' * 200)[:len_sentence])
      train_y.append(
        int(line.replace('\ufeff', '').replace('\n', '').split('\t')[0]))
      raw_text.append(line.replace('\ufeff', '').replace('\n', '').split('\t'))
  word_dict = {}
  word_count = {}
  index = 0
  for line in train_x:
    word_separate = []
    for idx in range(len(line)):
      word_separate.append(line[idx])
    word_separate = sorted(list(set(word_separate)))
    # sorted for n-gram concurrence added
    for char in word_separate:
      if char not in word_count:
        word_count[char] = 1
      else:
        word_count[char] += 1
      if char not in word_dict and word_count[char] > low_rate:
        word_dict[char] = index
        index += 1
  dict_length = len(word_dict)
  print('length of wordDict:', len(word_dict))
  train_output_x = _embedding(train_x, word_dict)
  del train_x
  train_output_y = _one_hot(train_y)
  del train_y
  return train_output_x, train_output_y, dict_length, word_dict, raw_text

def _embedding(data, word_dict):
  data_output = []
  for line in data:
    word_separate = []
    for idx in range(len(line)):
      word_separate.append(line[idx])
    embedding_word = []
    for char in word_separate:
      if char in word_dict.keys():
        embedding_word.append(word_dict[char])
      else:
        embedding_word.append(len(word_dict))
    data_output.append(np.array(embedding_word))
  data_output = np.array(data_output)
  return data_output

def _one_hot(data):
  largest = max(data)
  out = []
  for idx in range(len(data)):
    temp = np.zeros(largest + 1)
    temp[data[idx]] += 1
    out.append(temp)
  out = np.array(out)
  return out
