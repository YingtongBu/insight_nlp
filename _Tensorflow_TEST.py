#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Common import *
from Tensorflow import *

if __name__ == "__main__":
  parser = OptionParser(usage="cmd dev1@dir1 dir2")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  #default = False, help = "")
  (options, args) = parser.parse_args()

  sess = tf.Session()
  x = tf.constant([[0.,],
                   [1.,]])
  y = tf.constant([[2.,],
                   [3.,]])
  output = log_sum([x, y])
  print(sess.run(output))

  
