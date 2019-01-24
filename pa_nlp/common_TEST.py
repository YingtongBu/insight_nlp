#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from pa_nlp.common import *

if __name__ == "__main__":
  parser = OptionParser(usage="cmd dev1@dir1 dir2")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
  #default = False, help = "")
  (options, args) = parser.parse_args()

  print(uniq(["summer", "rain", "rain", 2, 2, 3, 1, "rain", "rain"]))

  data = create_list([10])
  data[0] = 1
  print(data)

  data = create_list([3, 4], None)
  data[0][0] = 1
  print(data)

  data = create_list([3, 4, 5], None)
  data[0][0][0] = 1
  print(data)
  
  print(get_module_path("NLP.translation.Translate"))

  print(eq(0, 0, EPSILON))
  print(eq(1.2345678912345678e30, 1.23456789123456689e30, 1e-13))

  path = "."
  for full_name in get_files_in_folder(path, ["py"], True):
    is_existed = os.path.exists(full_name)
    print(f"{full_name}, {is_existed}")
    assert is_existed


  data = [("a", 1), ("a", 2), ("b", 3), ("c", 4)]
  print(list(group_by_key(iter(data))))

  dists = [1, 2, 3, 4]
  print(collections.Counter([discrete_sample(dists) for freq in range(100000)]))
