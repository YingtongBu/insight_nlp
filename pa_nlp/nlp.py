#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

from pa_nlp import *

INF         = float("inf")
EPSILON     = 1e-6

def next_k(data: typing.Iterator, k: int):
  _ = range(k)
  data_iter = iter(data)
  while True:
    buff = list(zip(_, data_iter))
    if buff == []:
      break
    batch_data = list(map(itemgetter(1), buff))
    yield batch_data

def ensure_random_seed_for_one_time(buff={}):
  key = "randomized"
  status = buff.get(key, False)
  if not status:
    random.seed()
    buff[key] = True

def get_file_line_count(file_name: str):
  return int(os.popen(f"wc -l {file_name}").read().split()[0])

def get_files_line_count(file_names: list):
  return sum([get_file_line_count(f) for f in file_names])

def get_new_temporay_file():
  return tempfile.NamedTemporaryFile(delete=False).name

def next_line_from_file(file_name: str, max_count: int=-1):
  for idx, ln in enumerate(open(file_name)):
    if (max_count > 0 and idx < max_count) or max_count <= 0:
      yield ln.rstrip()

def next_line_from_files(file_names: list, max_count: int=-1):
  for f in file_names:
    yield from next_line_from_file(f, max_count)

def replace_list(data: list, func):
  return [func(d) for d in data]

def segment_contain(seg1: list, seg2: list):
  if seg1[0] <= seg2[0] and seg2[1] <= seg1[1]:
    return 1
  if seg2[0] <= seg1[0] and seg1[1] <= seg2[1]:
    return -1
  return 0

def segment_intersec(seg1: list, seg2: list):
  return not segment_no_touch(seg1, seg2)

def segment_no_touch(seg1: list, seg2: list):
  return seg1[1] <= seg2[0] or seg2[1] <= seg1[0]

def uniq(data: list)->typing.Iterator:
  '''
  :param data: must be sorted.
  '''
  prev = None
  for d in data:
    if prev is None or d != prev:
      yield d
      prev = d

def norm1(vec):
  vec = array(vec)
  nm = float(sum(abs(vec)))
  return vec if eq(nm, 0) else vec / nm

def norm2(vec):
  vec = array(vec)
  nm = math.sqrt(sum(vec * vec))
  return vec if eq(nm, EPSILON) else vec / nm

def cmp(a, b)-> int:
  return (a > b) - (a < b)

def get_home_dir():
  return os.environ["HOME"]

def ensure_folder_exists(folder: str, delete_first: bool=False)-> None:
  if delete_first:
    execute_cmd(f"rm -r {folder}")

  if not os.path.exists(folder):
    execute_cmd(f"mkdir {folder}")

def split_to_sublist(data)-> list:
  '''
  :param data: [[a1, b1, ...], [a2, b2, ...], ...]
  :return: [a1, a2, ...], [b1, b2, ...]
  '''
  if data == []:
    return []
  
  size = len(data[0])
  result = [[] for _ in range(size)]
  for tuple in data:
    for pos in range(size):
      result[pos].append(tuple[pos])
  
  return result
  
def get_module_path(module_name)-> typing.Union[str, None]:
  '''
  This applys for use-defined moudules.
  e.g., get_module_path("NLP.translation.Translate")
  '''
  module_name = module_name.replace(".", "/") + ".py"
  for path in sys.path:
    path = path.strip()
    if path == "":
      path = os.getcwd()
    
    file_name = os.path.join(path, module_name)
    if os.path.exists(file_name):
      return path
  
  return None

def norm_regex(regexExpr)-> str:
  return regexExpr\
    .replace("*", "\*")\
    .replace("+", "\+")\
    .replace("?", "\?")\
    .replace("[", "\[").replace("]", "\]")\
    .replace("(", "\(").replace(")", "\)")\
    .replace("{", "\{").replace("}", "\}")\
    .replace(".", "\.")

def pydict_file_read(file_name, max_num: int=-1)-> typing.Iterator:
  assert file_name.endswith(".pydict")
  data_num = 0
  with open(file_name) as fin:
    for idx, ln in enumerate(fin):
      if max_num >= 0 and idx + 1 > max_num:
        break
      if idx > 0 and idx % 10_000 == 0:
        print_flush(f"{file_name}: {idx} lines have been loaded.")

      try:
        obj = eval(ln)
        yield obj
        data_num += 1

      except Exception as err:
        print(f"ERR in reading {file_name}:{idx + 1}: {err} '{ln}'")

  print(f"{file_name}: #data={data_num}")

def pydict_file_write(data: typing.Iterator, file_name)-> None:
  assert file_name.endswith(".pydict")
  with open(file_name, "w") as fou:
    for obj in data:
      obj_str = str(obj)
      if "\n" in obj_str:
        print(f"ERR: in write_pydict_file: not '\\n' is allowed: '{obj_str}'")
      print(obj, file=fou)

def get_file_extension(file_name: str)-> str:
  return file_name.split(".")[-1]

def replace_file_name(file_name: str, old_suffix: str, new_suffix: str):
  assert old_suffix in file_name
  return file_name[: len(file_name) - len(old_suffix)] + new_suffix

def get_files_in_folder(data_path, file_extensions: list=None,
                        resursive=False)-> list:
  '''file_exts: should be a set, or None, e.g, ["wav", "flac"]
  return: a list, [fullFilePath]'''
  def legal_file(short_name):
    if short_name.startswith("."):
      return False
    ext = get_file_extension(short_name)
    return is_none_or_empty(file_extensions) or ext in file_extensions

  if file_extensions is not None:
    assert isinstance(file_extensions, (list, dict))
    file_extensions = set(file_extensions)

  for path, folders, files in os.walk(data_path, resursive):
    for short_name in files:
      if legal_file(short_name):
        yield os.path.realpath(os.path.join(path, short_name))

def split_data_by_func(data, func):
  data1, data2 = [], []
  for d in data:
    if func(d):
      data1.append(d)
    else:
      data2.append(d)
  return data1, data2

def is_none_or_empty(data)-> bool:
  '''This applies to any data type which has a __len__ method'''
  if data is None:
    return True
  if isinstance(data, (str, list, set, dict)):
    return len(data) == 0
  return False

def to_readable_time(seconds: float):
  result = []
  days = int(seconds // (3600 * 24))
  if days > 0:
    result.append(f"{days} d")
    seconds -= days * 3600 * 24

  hours = int(seconds // 3600)
  if hours > 0:
    result.append(f"{hours} h")
    seconds -= 3600 * hours

  minutes = int(seconds // 60)
  if minutes > 0:
    result.append(f"{minutes} m")
    seconds -= 60 * minutes

  if seconds > 0:
    result.append(f"{seconds:.3} s")

  return " ".join(result)

def get_log_time():
  return time.strftime('%x %X')

def execute_cmd(*cmds)-> int:
  cmd = " ".join(cmds)
  start = time.time()
  print_flush(f"{get_log_time()} [start] executing '{cmd}'")

  ret = os.system(cmd)
  status = "OK" if ret == 0 else "fail"
  duration = time.time() - start
  readable_time = to_readable_time(duration)
  print_flush(f"{get_log_time()} [{status}] {readable_time}, executing '{cmd}'")
  return ret

def to_utf8(line)-> typing.Union[str, None]:
  if type(line) is str:
    try:
      return line.encode("utf8")
    except:
      print("Warning: in toUtf8(...)")
      return None
    
  elif type(line) is bytes:
    return line
  
  print("Error: wrong type in toUtf8(...)")
  return None

def print_flush(cont, stream=None)-> None:
  if stream is None:
    stream = sys.stdout
  print(cont, file=stream)
  stream.flush()

def eq(v1, v2, prec=EPSILON):
  return abs(v1 - v2) < prec

def get_memory(size_type="rss")-> float:
  '''Generalization; memory sizes (MB): rss, rsz, vsz.'''
  content = os.popen(f"ps -p {os.getpid()} -o {size_type} | tail -1").read()
  return round(float(content) / 1024, 3)

def discrete_sample(dists)-> int:
  '''each probability must be greater than 0'''
  dists = array(dists)
  assert all(dists >= 0)
  accsum = scipy.cumsum(dists)
  expNum = accsum[-1] * random.random()
  return bisect.bisect(accsum, expNum)

def log_sum(ds):
  '''input: [d1, d2, d3..] = [log(p1), log(p2), log(p3)..]
      output: log(p1 + p2 + p3..)
  '''
  dv = max(ds)
  e = math.log(sum([math.exp(d - dv) for d in ds]))
  return dv + e

#depricated. Use tensorflow
def log_f_prime(fss, weight):
  '''input: fss: a list of feature vectors
      weight: scipy.array
      output: return log-gradient of log-linear model.
  '''
  #dn  = logsum([(ws.T * f)[0] for f in fs])
  dn  = log_sum(list(map(weight.dot, fss)))
  pdw = array([0.0] * len(weight))
  for fs in fss:
    pdw += math.exp(weight.dot(fs) - dn) * fs
  return pdw

def group_by_key_fun(data, key_fun=None):
  '''Note, the spark.group_by_key requires the data is sorted by keys.
  @:return a dict
  '''
  result = collections.defaultdict(list)
  for d in data:
    key = d[0] if key_fun is None else key_fun(d)
    result[key].append(d)

  return result

def create_batch_iter_helper(title: str, data: list, batch_size: int,
                             epoch_num: int, shuffle=True):
  '''
  :param data: [[word-ids, label], ...]
  :return: iterator of batch of [words-ids, label]
  '''
  for epoch_id in range(epoch_num):
    if shuffle:
      random.shuffle(data)
      
    next = iter(data)
    _ = range(batch_size)
    while True:
      batch = list(map(itemgetter(1), zip(_, next)))
      if batch == []:
        break
        
      samples = list(map(itemgetter(0), batch))
      labels = list(map(itemgetter(1), batch))
      yield samples, labels
     
    print(f"The '{title}' {epoch_id + 1}/{epoch_num} epoch has finished!")

def create_batch_iter_helper1(title: str, data: list, batch_size: int,
                             epoch_num: int, shuffle=True):
  '''
  :param data: [[word-ids, label, other1, other2], ...]
  :return: iterator of batch of [words-ids, label, other1, other2]
  '''
  for epoch_id in range(epoch_num):
    if shuffle:
      random.shuffle(data)

    next = iter(data)
    _ = range(batch_size)
    while True:
      batch = list(map(itemgetter(1), zip(_, next)))
      if batch == []:
        break

      yield split_to_sublist(batch)

    print(f"The '{title}' {epoch_id + 1}/{epoch_num} epoch has finished!")

def display_server_info():
  import socket
  host_name = socket.gethostname()
  ip = socket.gethostbyname(host_name)
  print(f"server information: {host_name}: {ip}, process: {os.getpid()}")

