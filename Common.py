#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from collections import defaultdict, namedtuple, Counter
from operator import methodcaller, attrgetter, itemgetter, add
from optparse import OptionParser
from scipy import array

import bisect
import collections
import copy
import datetime
import functools
import heapq
import itertools
import logging
import math
import multiprocessing as mp
import numpy as np
import operator
import optparse
import os
import pickle
import pprint
import queue
import random
import re
import scipy
import struct
import sys
import time

INF         = float("inf")
EPSILON     = 1e-6

def split_to_sublist(data):
  '''
  :param data: [[a1, b1, ...], [a2, b2, ...], ...]
  :return: [a1, a2, ...], [b1, b2, ...]
  '''
  if data == []:
    return []
  
  size = len(data[0])
  result = [[] for _ in range(size)]
  for tuple in data:
    for pos in range(size):
      result[pos].append(tuple[pos])
  
  return result
  
def get_module_path(module_name):
  '''
  e.g., get_module_path("NLP.Translation.Translate")
  '''
  module_name = module_name.replace(".", "/") + ".py"
  for path in sys.path:
    path = path.strip()
    if path == "":
      path = os.getcwd()
    
    file_name = os.path.join(path, module_name)
    if os.path.exists(file_name):
      return path
  
  return None

def norm_regex(regexExpr):
  return regexExpr\
    .replace("*", "\*")\
    .replace("+", "\+")\
    .replace("?", "\?")\
    .replace("[", "\[").replace("]", "\]")\
    .replace("(", "\(").replace(")", "\)")\
    .replace("{", "\{").replace("}", "\}")\
    .replace(".", "\.")

def read_pydict_file(file_name):
  assert file_name.endswith(".pydict")
  data = []
  for idx, ln in enumerate(open(file_name)):
    try:
      obj = eval(ln)
      data.append(obj)
    except:
      print(f"ERR in reading {file_name}:{idx + 1}: '{ln}'")
      
  return data

def write_pydict_file(data: list, file_name):
  with open(file_name, "w") as fou:
    for obj in data:
      obj_str = str(obj)
      if "\n" in obj_str:
        print(f"ERR: in write_pydict_file: not '\\n' is allowed: '{obj_str}'")
      print(obj, file=fou)

def get_files_in_folder(data_path, file_exts=None, resursive=False):
  '''
    do NOT set fileExt=html, as in some rare cases, all data files do not have
    an file extension.
    return: an iterator, [fullFilePath]
  '''
  def get_extension(short_name):
    toks = short_name.split(".")
    return short_name if len(toks) == 1 else toks[-1]
  
  for path, folders, files in os.walk(data_path):
    for short_name in files:
      if short_name.startswith("."):
        continue
      ext = get_extension(short_name)
      if not is_none_or_empty(file_exts) and ext not in file_exts:
        continue
      
      yield os.path.realpath(os.path.join(path, short_name))
      if not resursive:
        break

def create_list(shape: list, value=None):
  assert len(shape) > 0
  if len(shape) == 1:
    return [value for _ in range(shape[0])]
  else:
    return [create_list(shape[1:], value) for _ in range(shape[0])]

def split_by(data, func):
  data1, data2 = [], []
  for d in data:
    if func(d):
      data1.append(d)
    else:
      data2.append(d)
  return data1, data2

def is_none_or_empty(data):
  '''This applies to any data type which has a __len__ method'''
  if data is None:
    return True
  if isinstance(data, (str, list, set, dict)):
    return len(data) == 0
  return False

def execute_cmd(cmd):
  ret = os.system(cmd)
  status = "OK" if ret == 0 else "fail"
  date = time.strftime('%x %X')
  print(f"{date} [{status}] executing '{cmd}'")
  sys.stdout.flush()

def to_utf8(line):
  if type(line) is str:
    try:
      return line.encode("utf8")
    except:
      print("Warning: in toUtf8(...)")
      return None
    
  elif type(line) is bytes:
    return line
  
  print("Error: wrong type in toUtf8(...)")
  return None

def print_with_flush(cont, stream=None):
  if stream is None:
    stream = sys.stdout
  print(cont, file=stream)
  stream.flush()

def get_installed_packages():
  import pip
  packages = pip.get_installed_distributions()
  packages = sorted(["%s==%s" % (i.key, i.version) for i in packages])
  return packages

def eq(v1, v2, prec=EPSILON):
  return abs(v1 - v2) < prec

def get_memory(size_type="rss"):
  '''Generalization; memory sizes (MB): rss, rsz, vsz.'''
  content = os.popen(f"ps -p {os.getpid()} -o {size_type} | tail -1").read()
  return round(content / 1024, 3)

def discrete_sample(dists):
  '''each probability must be greater than 0'''
  dists = array(dists)
  assert all(dists >= 0)
  accsum = scipy.cumsum(dists)
  expNum = accsum[-1] * random.random()
  return bisect.bisect(accsum, expNum)

def log_sum(ds):
  '''input: [d1, d2, d3..] = [log(p1), log(p2), log(p3)..]
      output: log(p1 + p2 + p3..)
  '''
  dv = max(ds)
  e = math.log(sum([math.exp(d - dv) for d in ds]))
  return dv + e

def log_f_prime(fss, weight):
  '''input: fss: a list of feature vectors
      weight: scipy.array
      output: return log-gradient of log-linear model.
  '''
  #dn  = logsum([(ws.T * f)[0] for f in fs])
  dn  = log_sum(list(map(weight.dot, fss)))
  pdw = array([0.0] * len(weight))
  for fs in fss:
    pdw += math.exp(weight.dot(fs) - dn) * fs
  return pdw

def create_batch_iter_helper(title: str, data, batch_size, epoch_num,
                             shuffle=True):
  '''
  :param data: [[word-ids, label], ...]
  :return: iterator of batch of [words-ids, label]
  '''
  for epoch_id in range(epoch_num):
    if shuffle:
      random.shuffle(data)
      
    next = iter(data)
    _ = range(batch_size)
    while True:
      batch = list(map(itemgetter(1), zip(_, next)))
      if batch == []:
        break
        
      samples = list(map(itemgetter(0), batch))
      labels = list(map(itemgetter(1), batch))
      yield samples, labels
     
    print(f"The '{title}' {epoch_id + 1}/{epoch_num} epoch has finished!")

