#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Insight_NLP.Common import *

if __name__ == "__main__":
  parser = OptionParser(usage="cmd dev1@dir1 dir2")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  #default = False, help = "")
  (options, args) = parser.parse_args()

  data = create_list([10])
  data[0] = 1
  print(data)

  data = create_list([3, 4], None)
  data[0][0] = 1
  print(data)

  data = create_list([3, 4, 5], None)
  data[0][0][0] = 1
  print(data)
  
  print(get_module_path("NLP.Translation.Translate"))
  
  path = "."
  for full_name in get_files_in_folder(path, ["py"], True):
    is_existed = os.path.exists(full_name)
    print(f"{full_name}, {is_existed}")
    assert is_existed
