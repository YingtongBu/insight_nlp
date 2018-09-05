from Insight_NLP.Common import *
from Insight_NLP.Vocabulary import *

'''The format of data is defined as
1. each line is a python dict string, denoting a sample dict.
2. each sample dict has basic "label" and "text" keys, whose types are "int" and
"string" respectively.
3. In the function "normalize_data_file", a new key "word_list" would be added.
4. In the function "create_batch_iterator", a new key "word_ids" would be added,
which would be fed in a DL model.
'''

EMPTY_TOKEN = "<empty>"
OOV_TOKEN   = "<oov>"

def create_batch_iterator(data_file, batch_size, max_seq_length,
                          epoch_num, vob: Vocabulary,
                          remove_OOV: bool, shuffle: bool):
  def read_buffer():
    samples = read_pydict_file(data_file)
    for sample in enumerate(samples):
      sample["word_ids"] = \
        vob.convert_to_word_ids(sample["word_list"],
                                remove_OOV=remove_OOV,
                                mark_OOV=OOV_TOKEN,
                                output_length=max_seq_length,
                                mark_empty=EMPTY_TOKEN)
    return samples
  
  samples = read_buffer()
  
  return create_batch_iter_helper(samples, batch_size, epoch_num, shuffle)

def create_vocabulary(train_file, min_freq, out_vob_file):
  data = read_pydict_file(train_file)
  words_list = [sample["text"].split() for sample in data]
  vob = Vocabulary()
  vob.create_vob_from_data(words_list, min_freq)
  vob.add_word(EMPTY_TOKEN)
  vob.add_word(OOV_TOKEN)
  vob.save_to_file(out_vob_file)
  
def normalize_data_file(file_name, out_file_name):
  data = read_pydict_file(file_name)
  for sample in data:
    text  = sample["text"]
    sample["word_list"] = normalize_text(text).split()
   
  write_pydict_file(data, out_file_name)

def normalize_text(string: str):
  """
  Tokenization/string cleaning for Chinese and English mixed data
  """
  string = re.sub(r"[^A-Za-z0-9\u4e00-\u9fa5()（）！？，,!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  # clean for chinese character
  new_string = ""
  for char in string:
    if re.findall(r"[\u4e00-\u9fa5]", char) != []:
      char = " " + char + " "
    new_string += char
    
  return new_string.strip().lower().replace('\ufeff', '')

