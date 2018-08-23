#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from collections import defaultdict, namedtuple, Counter
from operator import methodcaller, attrgetter, itemgetter, add
from optparse import OptionParser

import bisect
import collections
import copy
import datetime
import heapq
import itertools
import logging
import functools
import math
import multiprocessing
import operator
import optparse
import os
import pickle
import pprint
import queue
import random
import re
import struct
import sys
import time

INF         = float("inf")
EPSILON     = 1e-6

try:
  import scipy
  from scipy import array
except ImportError:
  print("Does not find package 'scipy'")
  
def norm_regex(regexExpr):
  return regexExpr\
    .replace("*", "\*")\
    .replace("+", "\+")\
    .replace("?", "\?")\
    .replace("[", "\[").replace("]", "\]")\
    .replace("(", "\(").replace(")", "\)")\
    .replace("{", "\{").replace("}", "\}")\
    .replace(".", "\.")

def read_CSV(fname, splitter="\t", col_num=None):
  ''' All column name would be removed.
  '''
  def normalize(toks):
    result = []
    for tok in toks:
      p = tok.find("=")
      result.append(tok[p + 1:].strip())
    return result
  
  table = []
  with open(fname) as fin:
    for ln in fin:
      toks = ln.split(splitter)
      toks = normalize(toks)
      if col_num is None:
        table.append(toks)
      else:
        size = len(toks)
        if size != col_num:
          print(f"Warn: {size} != {col_num}: {toks}")
          continue
        else:
          table.append(toks)
  
  return table

def write_CSV(fname, data, col_names=None, splitter="\t", col_num=None):
  '''We could save name for each column, which is more human-readable.
   data: 2D array
  '''
  assert len(col_names) == col_num
  with open(fname) as fou:
    for row in data:
      if col_num is not None:
        assert len(row) == col_num, row
        
      if col_names is not None:
        assert len(row) == len(col_names), row
        rowStr = [name + "=" + str(e) for name, e in zip(col_names, row)]
      else:
        rowStr = [str(e) for e in row]
      print(splitter.join(rowStr), file=fou)

def get_files_in_folder(data_path, file_exts=None, resursive=False):
  '''
    do NOT set fileExt=html, as in some rare cases, all data files do not have
    an file extension.
    return: an iterator, [[fullFilePath, fileID] ...]
  '''
  for root, subFolders, files in os.walk(data_path):
    for fname in files:
      if fname.startswith("."):
        continue
      ext = fname.split(".")[0]
      if not is_none_or_empty(file_exts) and ext not in file_exts:
        continue
      
      yield os.path.join(data_path, fname), fname.split(".")[0]
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

if __name__ == "__main__":
  parser = OptionParser(usage="cmd dev1@dir1 dir2")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  #default = False, help = "")
  (options, args) = parser.parse_args()

  data = create_list([10])
  data[0] = 1
  print(data)

  data = create_list([3, 4], None)
  data[0][0] = 1
  print(data)

  data = create_list([3, 4, 5], None)
  data[0][0][0] = 1
  print(data)
