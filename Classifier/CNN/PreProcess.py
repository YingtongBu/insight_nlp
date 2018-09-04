#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

import re
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
  return [x_text, y]

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
  # clean for chinese character
  new_string = ""
  for char in string:
    if re.findall(r"\u4e00-\u9fa5", char) != []:
      char = " " + char + " "
    new_string += char
  return new_string.strip().lower().replace('\ufeff', '').split(" ")
