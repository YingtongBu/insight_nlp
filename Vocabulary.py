#!/usr/bin/env python

from Insight_NLP.Common import *

class Vocabulary:
  def __init__(self):
    self._clean()
    
  def _clean(self):
    self._word2freq = defaultdict(int)
    self._word2Id = {}
    self._words = []
    
  def create_vob_from_data(self, data_list: list, min_freq=None):
    '''
    data_list: a list of token list.
    For example, [["welcome", "to", "ping'an"], ["欢迎", "来到", "平安"],
                  ["欢", "迎", "来", "到", "平安"]]
    '''
    self._clean()
    counter = Counter()
    for tokens in data_list:
      counter.update(tokens)

    for word, freq in counter.most_common(len(counter)):
      if min_freq is not None and freq < min_freq:
        break
      self.add_word(word)
      self._word2freq[word] = freq
    
  def save_to_file(self, file_name):
    '''
    each line: word idx freq
    '''
    with open(file_name, "w") as fou:
      for word in self._words:
        idx = self.get_word_id(word)
        freq = self._word2freq[word]
        print(f"{word} {idx} {freq}", file=fou)
        
  def load_from_file(self, file_name):
    ''' The first word each line would be read.
    '''
    self._clean()
    for ln in open(file_name):
      self.add_word(ln.split()[0])
      
    print(f"loaded {self.size()} words from '{file_name}'.")

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
  
  def convert_to_word_ids(self, words: list,
                          remove_OOV=True, mark_OOV=None,
                          output_length=None, mark_empty=None):
    '''
    if remove_OOV is False, then empty_mark should be "special position holder"
    which has been added into vob.
    '''
    if not remove_OOV:
      assert mark_OOV is not None, "you must set 'mark_OOV'"
      id_OOV = self.get_word_id(mark_OOV)
      assert id_OOV is not None, "you must set 'mark_OOV' in vob."
    
    if output_length is not None:
      assert mark_empty is not None, "You must set 'mark_empty'."
      id_empty = self.get_word_id(mark_empty)
      assert id_empty is not None, "You must set 'mark_empty' in vob."
      
    ids = [self.get_word_id(word) for word in words]
    if remove_OOV:
      ids = [id for id in ids if id is not None]
    else:
      ids = [id if id is not None else id_OOV for id in ids]
      
    if output_length is not None:
      ids = ids[: output_length]
      ids.extend([id_empty] * (output_length - len(ids)))
      
    return ids
