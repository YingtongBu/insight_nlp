#!/usr/bin/env python
#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import re
from sklearn import preprocessing

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
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
  return string.strip().lower()

def load_data_eng(file_path, num_of_class):
  x_text, trainY, y = [], [], []
  data = open(file_path, 'r', encoding='latin').readlines()[1:]
  data = [sample.strip() for sample in data]
  for row in data:
    row = row.split('\t')
    x_text.append(row[1].replace('\ufeff', ''))
    trainY.append(row[0])
  # Split by words
  x_original = x_text
  x_text = [clean_str(sent) for sent in x_text]
  # clean y
  for item in trainY:
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

if __name__ == '__main__':
  pass
