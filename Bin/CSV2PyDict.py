#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

import csv
import os
import optparse

def process(csv_file, pydict_file):
  data_list = []
  with open(csv_file)as f:
    data = csv.DictReader(f)
    titles = data.fieldnames
    for row in data:
      data_list.extend([{titles[i]: row[titles[i]] for i in range(len(titles))}])

  with open(pydict_file, "w") as output:
    for obj in data_list:
      print(obj, file=output)

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose", #default=False, help="")
  (options, args) = parser.parse_args()

  print(f"We assume that the first line of a csv is its titles")

  for csv_file in args:
    out_file = csv_file.replace(".csv", ".pydict")
    print(f"process {csv_file}")
    process(csv_file, out_file)

