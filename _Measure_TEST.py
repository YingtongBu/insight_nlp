# coding: utf8
# author: Tian Xia (summer.xia1@pactera.com)

from common import *
from measure import *

if __name__ == "__main__":
  parser = OptionParser(usage="cmd dev1@dir1 dir2")
  # parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  # default = False, help = "")
  (options, args) = parser.parse_args()
  
  print(Measure.calc_precision_recall_fvalue([0, 1, 1, 2],
                                             [0, 0, 1, 2]))

