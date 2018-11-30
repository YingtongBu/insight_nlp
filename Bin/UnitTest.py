#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from Common import *

def get_testing_files(cwd, args):
  '''
  :return: [full-path-file-name]
  '''
  if not is_none_or_empty(args):
    for short_file in args:
      yield os.path.join(cwd, short_file)

  else:
    file_names = []
    for file_name in get_files_in_folder(".", ["py"], True):
      if file_name.endswith("TEST.py"):
        base_name = os.path.basename(file_name)
        if not base_name.startswith("_"):
          assert False, f"Wrong file name: {file_name}"

        file_names.append(file_name)

    unit_list_file = "UnitTestList.data"
    if not os.path.exists(unit_list_file):
      print(f"You should set '{unit_list_file}' file, and put all scripts to "
            f"test into this file")
      exit(1)

    list_files = [os.path.realpath(name) for name in
                  open(unit_list_file).read().split()]

    set1 = set(file_names)
    set2 = set(list_files)
    buff = set1 - set2
    if len(buff) > 0:
      print(f"You missed {len(buff)} files!")
      print("\n".join(buff))
      exit(1)

    buff = set2 - set1
    if len(buff) > 0:
      print(f"You have extra {len(buff)} files!")
      print("\n".join(buff))
      exit(1)

    yield from file_names

def execute_testing_file(cwd: str, full_file_name: str):
  title = ">" * 32 + f"Testing {full_file_name}" +  ">" * 32
  print(title)
  os.chdir("/tmp")
  execute_cmd(f"cp {full_file_name} .")

  short_file_name = os.path.basename(full_file_name)
  code = execute_cmd(f"python3 {short_file_name}")

  os.chdir(cwd)
  print("<" * len(title), "\n")

  return code

def main():
  parser = optparse.OptionParser(usage="cmd [optons] [file1.py file2.py ...]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  parser.add_option("--show", action="store_true",
                    help="show all scripts to test")
  (options, args) = parser.parse_args()

  cwd = os.getcwd()
  error_testings = []
  all_testing_files = list(get_testing_files(cwd, args))
  print(f"There are {len(all_testing_files)} testing files.")

  if options.show:
    print("\n".join(all_testing_files))
    return

  for full_file_name in all_testing_files:
    code = execute_testing_file(cwd, full_file_name)
    if code != 0:
      error_testings.append(full_file_name)

  print(f"{len(error_testings)} testing fail!")
  print("\n".join(error_testings))

if __name__ == "__main__":
  main()
