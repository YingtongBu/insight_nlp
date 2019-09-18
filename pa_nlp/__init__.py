#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from collections import defaultdict, namedtuple, Counter
from operator import methodcaller, attrgetter, itemgetter
from optparse import OptionParser

import abc
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
import struct
import sys
import time
import typing
import tempfile

try:
  import scipy
  from scipy import array
except ImportError:
  print(f"Err: can not import 'scipy'")

