#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from common import *
from measure import *
import common as nlp

if __name__ == "__main__":
  parser = OptionParser(usage="cmd dev1@dir1 dir2")
  # parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  # default = False, help = "")
  (options, args) = parser.parse_args()
  
  print(Measure.calc_precision_recall_fvalue([0, 1, 1, 2],
                                             [0, 0, 1, 2]))

  refs = ["who is there".split(), "who is there".split()]
  hyps = ["is there".split(), "".split()]
  assert nlp.eq(Measure.WER(refs, hyps), 2 / 3)

  time_start = time.time()
  ref = "who is there in the playground , Summer Rain, can you see it".split()
  hyp = "who is there in playground , Summer Rain, can you see it".split()
  refs = [ref] * 100
  hyps = [hyp] * 100
  print(Measure.WER(refs, hyps, True))
  duration = time.time() - time_start
  print(f"[time: {duration}, memory: {nlp.get_memory()}")
