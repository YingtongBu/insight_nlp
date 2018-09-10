#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Insight_NLP.Common import *

'''This class has added two special tokens, <empty> and <oov> by default.
'''
EMPTY_TOKEN = "<empty>"
OOV_TOKEN   = "<oov>"

class Vocabulary:
  def __init__(self, remove_OOV: bool, output_length: int):
    '''
    :param remove_OOV:
    :param output_length: int, or None
    '''
    self._remove_OOV = remove_OOV
    self._output_length = output_length
    self._clear()
    
  def _update_special_tokens(self):
    self.add_word(EMPTY_TOKEN)
    self.add_word(OOV_TOKEN)
    self.id_EMPTY_TOKEN = self.get_word_id(EMPTY_TOKEN)
    self.id_OOV_TOKEN = self.get_word_id(OOV_TOKEN)
    
  def _clear(self):
    self._word2freq = defaultdict(int)
    self._word2Id = {}
    self._words = []
    
  def create_vob_from_data(self, data_list: list, min_freq=None):
    '''
    data_list: a list of token list.
    For example, [["welcome", "to", "ping'an"], ["欢迎", "来到", "平安"],
                  ["欢", "迎", "来", "到", "平安"]]
    '''
    self._clear()
    
    counter = Counter()
    for tokens in data_list:
      counter.update(tokens)

    for word, freq in counter.most_common(len(counter)):
      if min_freq is not None and freq < min_freq:
        break
      self.add_word(word)
      self._word2freq[word] = freq
      
    self._update_special_tokens()
    
  def save_model(self):
    '''
    each line: word idx freq
    '''
    with open("vob.data", "w") as fou:
      for word in self._words:
        idx = self.get_word_id(word)
        freq = self._word2freq[word]
        print(f"{word} {idx} {freq}", file=fou)
        
  def load_model(self):
    ''' The first word each line would be read.
    '''
    self._clear()
    for ln in open("vob.data"):
      self.add_word(ln.split()[0])

    self._update_special_tokens()
    
    print(f"loaded {self.size()} words from vob.data.")

  def add_word(self, word):
    ''' add word if it does not exist, and then return its id.
    '''
    self._word2freq[word] += 1
    idx = self._word2Id.get(word, None)
    if idx is not None:
      return idx
    self._words.append(word)
    self._word2Id[word] = len(self._word2Id)
    return len(self._words) - 1

  def get_word_id(self, word):
    return self._word2Id.get(word, None)

  def get_word(self, idx):
    return self._words[idx] if 0 <= idx < len(self._words) else None

  def size(self):
    return len(self._words)
  
  def convert_to_word_ids(self, words: list):
    ids = [self.get_word_id(word) for word in words]
    if self._remove_OOV:
      ids = [id for id in ids if id is not None]
    else:
      ids = [id if id is not None else self.id_OOV_TOKEN for id in ids]
      
    if self._output_length is not None:
      ids = ids[: self._output_length]
      ids.extend([self.id_EMPTY_TOKEN] * (self._output_length - len(ids)))
      
    return ids
