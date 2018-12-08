#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from web.search_engine import SearchEngine

if __name__ == '__main__':
  searcher = SearchEngine()
  result = searcher.search('拼多多', item_nums=100)
  print(result)
