from bs4 import BeautifulSoup

file_object = open('/Users/xinjin/Desktop/dataset/round1_train_20180518/增减持/html/20596892.html')
html_context = file_object.read()
soup = BeautifulSoup(html_context, 'html5lib')
trs = soup.find_all('tr')
print(trs[0])
# for tr in soup.find_all('tr'):
#     print(tr.find_all('td'))