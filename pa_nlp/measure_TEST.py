#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp.common import *
from pa_nlp.measure import *
from pa_nlp import common as nlp

if __name__ == "__main__":
  parser = OptionParser(usage="cmd dev1@dir1 dir2")
  # parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  # default = False, help = "")
  (options, args) = parser.parse_args()
  
  print(Measure.calc_precision_recall_fvalue([0, 1, 1, 2],
                                             [0, 0, 1, 2]))

  refs = ["who is there", "who is there"]
  hyps = ["is there", ""]
  assert nlp.eq(Measure.calc_WER(refs, hyps), 2 / 3)

  time_start = time.time()
  ref = "who is there in the playground , Summer Rain, can you see it"
  hyp = "who is there in the playground , Summer Rain, can you see it"
  refs = [ref] * 100
  hyps = [hyp] * 100
  print(Measure.calc_WER(refs, hyps, True))
  duration = time.time() - time_start
  print(f"[time: {duration}, memory: {nlp.get_memory()}")
