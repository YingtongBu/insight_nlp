#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from Insight_NLP.Web.SearchEngine import SearchEngine

if __name__ == '__main__':
  searcher = SearchEngine(custom_search_key='005808576341306023160:yojc6z7o63u',
                      developer_key="AIzaSyC1o8pJAwMvaRugaRp9nWtvrGQs2_llEps")
  result = searcher.search('拼多多', item_nums=100)
  print(result)
