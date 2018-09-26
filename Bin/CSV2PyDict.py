#!/usr/bin/env python
#coding: utf8

'''
1. Suppose the first line of a csv is its titles.
'''

def process(csv_file, pydict_file):
  pass

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose", #default=False, help="")
  parser.add_option("--splitter", dest="splitter", default="\t", help="")
  (options, args) = parser.parse_args()

  for csv_file in args:
    out_file = csv_file.replace(".csv", ".pydict")
    print(f"process {csv_file}")
    process(csv_file, out_file) 

