#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
"""
This modules implements the scraper_wsj which is used to scrap news from Reuters
"""
import bs4
import datetime as dt
import logging
from Crawlers.Clean import clean_text
from NLP import news_title_filter, news_summary_filter

logger = logging.getLogger("NewsScraper.crawlers.wsj")

def scraper_wsj(driver, date):
  list_news = list()
  url = 'http://www.wsj.com/public/page/archive-' + \
        dt.datetime.strftime(date, '%Y-%m-%d') + '.html'
  try:
    driver.get(url)
    soup = bs4.BeautifulSoup(driver.page_source, 'lxml')
    for tag in soup.find('ul', {'class': 'newsItem'}).find_all('li'):
      title = clean_text(tag.find('a').get_text())
      if not news_title_filter(title):
        logger.debug('This news is not in interest based on title: {}'.format(
          title))
        continue
      this_news = dict()
      this_news['title'] = title
      this_news['url'] = tag.find('a').get('href')
      this_news['summary'] = clean_text(tag.find('p').get_text())
      if not news_summary_filter(this_news['summary']):
        logger.debug('This news is not in interest based on summary: {}'.format(
          this_news['summary']))
        continue
      try:
        driver.get(this_news['url'])
        art_soup = bs4.BeautifulSoup(driver.page_source, 'lxml')
        this_news['news_date'] = str(date)
        article = art_soup.find_all('p', {'class': None})
        texts = [a.get_text() for a in article if 'http' not in a.get_text()]
        text = ''.join(texts)
        this_news['article'] = clean_text(text)
      except:
        logger.exception("Error in scraping {}".format(this_news['url']))
      list_news.append(this_news)
  except:
    logger.exception("Error in scraping {}".format(url))
  return list_news

if __name__ == '__main__':
    pass