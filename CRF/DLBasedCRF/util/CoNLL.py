#!/usr/bin/env python3
#coding: utf8
#author: Xin Jin (xin.jin12@pactera.com)
from __future__ import print_function
import os

def conll_write(output_path, sentences, headers):
  if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
  f_out = open(output_path, 'w')
      
  for sentence in sentences:
    f_out.write("#")
    f_out.write("\t".join(headers))
    f_out.write("\n")
    for token_idx in range(len(sentence[headers[0]])):
      ace_data = [sentence[key][token_idx] for key in headers]
      f_out.write("\t".join(ace_data))
      f_out.write("\n")
    f_out.write("\n")
              
def read_co_nll(input_path, cols, comment_symbol=None, val_transformation=None):
  sentences = []    
  sentence_template = {name: [] for name in cols.values()}
  sentence = {name: [] for name in sentence_template.keys()}
  new_data = False
    
  for line in open(input_path):
    line = line.strip()
    if len(line) == 0 or (comment_symbol is not
                          None and line.startswith(comment_symbol)):
      if new_data:
        sentences.append(sentence)
        sentence = {name: [] for name in sentence_template.keys()}
        new_data = False
      continue
        
    splits = line.split()
    for col_idx, col_name in cols.items():
      val = splits[col_idx]
      if val_transformation is not None:
        val = val_transformation(col_name, val, splits)
      sentence[col_name].append(val)
            
    new_data = True
        
  if new_data:
    sentences.append(sentence)
            
  for name in cols.values():
    if name.endswith('_BIO'):
      iobes_name = name[0:-4] + '_class'

      class_name = name[0:-4] + '_class'
      for sentence in sentences:
        sentence[class_name] = []
        for val in sentence[name]:
          val_class = val[2:] if val != 'O' else 'O'
          sentence[class_name].append(val_class)

      iob_name = name[0:-4] + '_IOB'
      for sentence in sentences:
        sentence[iob_name] = []
        old_val = 'O'
        for val in sentence[name]:
          new_val = val
                    
          if new_val[0] == 'B':
            if old_val != 'I' + new_val[1:]:
              new_val = 'I' + new_val[1:]
                        
          sentence[iob_name].append(new_val)
          old_val = new_val

      iobes_name = name[0:-4] + '_IOBES'
      for sentence in sentences:
        sentence[iobes_name] = []
                
        for pos in range(len(sentence[name])):                    
          val = sentence[name][pos]
          if (pos + 1) < len(sentence[name]):
            next_val = sentence[name][pos + 1]
          else:
            next_val = 'O'

          new_val = val
          if val[0] == 'B':
            if next_val[0] != 'I':
              new_val = 'S' + val[1:]
          elif val[0] == 'I':
            if next_val[0] != 'I':
              new_val = 'E' + val[1:]

          sentence[iobes_name].append(new_val)
                   
  return sentences  