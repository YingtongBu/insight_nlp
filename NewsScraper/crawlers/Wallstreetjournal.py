#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
"""
This modules implements the scraper_wsj which is used to scrap news from Reuters
"""
import bs4
import datetime as dt
import logging
from crawlers.clean import cleanText
from NLP import newsTitleFilter, newsSummaryFilter

logger = logging.getLogger("NewsScraper.crawlers.wsj")

def scraperWsj(driver, date):
  listNews = list()
  url = 'http://www.wsj.com/public/page/archive-' + \
        dt.datetime.strftime(date, '%Y-%m-%d') + '.html'
  try:
    driver.get(url)
    soup = bs4.BeautifulSoup(driver.page_source, 'lxml')
    for tag in soup.find('ul', {'class': 'newsItem'}).find_all('li'):
      title = cleanText(tag.find('a').get_text())
      if not newsTitleFilter(title):
        logger.debug('This news is not in interest based on title: {}'.format(
          title))
        continue
      thisNews = dict()
      thisNews['title'] = title
      thisNews['url'] = tag.find('a').get('href')
      thisNews['summary'] = cleanText(tag.find('p').get_text())
      if not newsSummaryFilter(thisNews['summary']):
        logger.debug('This news is not in interest based on summary: {}'.format(
          thisNews['summary']))
        continue
      try:
        driver.get(thisNews['url'])
        art_soup = bs4.BeautifulSoup(driver.page_source, 'lxml')
        thisNews['news_date'] = str(date)
        article = art_soup.find_all('p', {'class': None})
        texts = [a.get_text() for a in article if 'http' not in a.get_text()]
        text = ''.join(texts)
        thisNews['article'] = cleanText(text)
      except:
        logger.exception("Error in scraping {}".format(thisNews['url']))
      listNews.append(thisNews)
  except:
    logger.exception("Error in scraping {}".format(url))
  return listNews

if __name__ == '__main__':
    pass