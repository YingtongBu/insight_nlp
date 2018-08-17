#coding: utf8
#author: Xinyi Wu (xinyi.wu5@pactera.com)
DELETE_WORDS = ['\n', '\t', '\xa0']

def removeWords(text_string, delete_words=DELETE_WORDS):
  for word in delete_words:
    text_string = text_string.replace(word, '')
  return text_string

def cleanText(text):
  return ' '.join(text.split()).lower()

if __name__ == '__main__':
    print(removeWords('\tThis \xa0 is a test.\n'))
    print(cleanText('Test clean Text Function.'))