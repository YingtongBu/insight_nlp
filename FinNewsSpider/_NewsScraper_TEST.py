#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

from FinNewsSpider.NewsScraper import Scraper

if __name__ == '__main__':
  news = Scraper('http://finance.eastmoney.com/news/1344,20180912944502216.html',
    language='zh')
  result = news.get_result()
  print(result)