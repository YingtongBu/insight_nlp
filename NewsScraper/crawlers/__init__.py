#coding: utf8
# author: Xinyi Wu (xinyi.wu5@pactera.com)
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
import datetime as dt
from crawlers.nytimes import scraperNytimes, scraperNytimesDaily
from crawlers.reuters import scraperReuters
from crawlers.wallstreetjournal import scraperWsj
import logging

logger = logging.getLogger("NewsScraper.crawlers")

class Scraper:
  """Base class for scrapers.
  """
  def __init__(self, phantomjsPath, username, password):
    logger.info("Initialize crawler")
    # store each news as a dict(title, url, newsDate, article, summary, time,
    #  location, people, event)
    self.listNews = list()

    # WSJ driver setting
    try:
      self.driver = webdriver.PhantomJS(executable_path=phantomjsPath)
      loginUrl = \
        'https://accounts.wsj.com/login?target=https%3A%2F%2Fwww.wsj.com'
      self.driver.get(loginUrl)
      usernameField = self.driver.find_element_by_id("username")
      passwordField = self.driver.find_element_by_id("password")
      loginButton = self.driver.find_element_by_class_name("basic-login-submit")
      usernameField.clear()
      passwordField.clear()
      usernameField.send_keys(username)
      passwordField.send_keys(password)
      loginButton.send_keys(Keys.ENTER)
      self.driver.wait = WebDriverWait(self.driver, 30)
      self.driver.wait.until(lambda driver:
                             driver.current_url == 'https://www.wsj.com/')
    except:
      logger.exception("Error in initializing WSJ crawler")

  def scraperReuters(self):
    logger.info("Start scraping Reuters")
    self.listNews = scraperReuters()
    logger.info('{} news scraped from Reuters'.format(len(self.listNews)))

  def scraperNytimes(self):
    logger.info("Start scraping NYTimes on World News page")
    self.listNews = scraperNytimes()
    logger.info('{} news scraped from NYTimes'.format(len(self.listNews)))

  def scraperNytimesDaily(self, date):
    logger.info("Start scraping from NYTimes for " +
                dt.datetime.strftime(date, '%Y-%m-%d'))
    self.listNews = scraperNytimesDaily(date)
    logger.info('{} news scraped from NYTimes for '.format(
      len(self.listNews)) + dt.datetime.strftime(date, '%Y-%m-%d'))

  def scraperWsj(self, date):
    logger.info("Start scraping Wall Street Journal for " +
                dt.datetime.strftime(date, '%Y-%m-%d'))
    self.listNews = scraperWsj(self.driver, date)
    logger.info('{} news scraped from Wall Street Journal for '.format(
      len(self.listNews)) + dt.datetime.strftime(date, '%Y-%m-%d'))

  def quit(self):
    self.driver.quit()

  @property
  def newsList(self):
    return self.listNews

if __name__ == '__main__':
    pass