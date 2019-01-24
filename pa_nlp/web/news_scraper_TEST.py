#!/usr/bin/env python
#coding: utf8
#author: Xinlu Yu (xinlu.yu1@pactera.com)

from pa_nlp.web.news_scraper import NewsScraper

if __name__ == '__main__':
  news = NewsScraper(
    'http://finance.eastmoney.com/news/1344,20180912944502216.html',
    language='zh')
  result = news.get_result()
  print(result)