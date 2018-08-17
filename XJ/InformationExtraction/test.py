# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import unicodedata
import os

read_path = '/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/FDDC_announcements_round1_test_a_20180605/重大合同/html/'
file_list = os.listdir(read_path)
document_dict = dict()

for file in file_list:
    file_object = open(read_path + file, 'r')    
    html_context = file_object.read()
    soup = BeautifulSoup(html_context, 'html5lib')
    sentence_list = []
    for content in soup.find_all('div', attrs={"type" : "content"}):
        content = unicodedata.normalize('NFKC', content.text.replace('\n', ' ').replace(' ', ''))
        if content != '':
            content_list = content.split('。')
            for sentence in content_list:
                if sentence != '':
                    sentence_list.append(sentence)
    document_dict[file] = sentence_list

test_list = []
for file in file_list:
    for item in document_dict[file]:
        test_list.append(file + '\t' + item)
    
file_write_obj = open('/Users/xinjin/MyLife/UCSD_Job/PATechUSResearchLab/Kensho/informationExtraction/dataset/test.txt', 'w')
for item in set(test_list):
    file_write_obj.writelines(item)
    file_write_obj.write('\n')
file_write_obj.close()
