#!/usr/bin/env python
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import os
import optparse
import sys

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  parser.add_option("--repoPath")
  (options, args) = parser.parse_args()
  
  real_path = os.path.join(os.getcwd(), sys.argv[0])
  # print(realPath)
  scipt_folder = os.path.split(real_path)[0]
  # print(sciptFolder)
  
  scipt = os.path.join(scipt_folder, "prepush_check.py")
  
  hook_dir = os.path.join(options.repoPath, ".git/hooks")
  os.system(f"cd {hook_dir};"
            f"rm pre-push;"
            f"cp {scipt} pre-push;"
            f"chmod u+x pre-push")
  
  print("Successful!")

