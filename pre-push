#!/usr/bin/env python3
#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import os
import re
import optparse
import sys
sys.path.append("")
print(sys.path)
from nlp.GitCheck.FormatChecker import *

USER_WHITE_LIST = set(["summer"])

def getAllCommitInfo():
  '''
  return [commit, email, [py-files]]
  '''
  
  def readBuffer():
    '''
    return {commit: {"email": author-email, "python": [py-files]}}
    '''
    try:
      ret = eval(open(".git/hooks/commit.data").read())
      return ret
    except:
      print(f"WARN: can not find .git/hooks/commit.data")
      return {}
    
  def writeBuffer(allCommitInfo):
    try:
      buffer = {}
      for commit, user, files in allCommitInfo:
        buffer[commit] = {"user": user, "python": files}
      
      print(buffer, file=open(".git/hooks/commit.data", "w"))
    except:
      print(f"WARN: can not write .git/hooks/commit.data")
    
  def getAllCommit():
    '''
    :return [commit, author-email] sorted by time.
    '''
    logs = os.popen("git log").read()
    commits = re.findall(r"^commit\s+(.*)$", logs, re.MULTILINE)
    emails  = re.findall(r"^Author:.*<(.*)@.*>", logs, re.MULTILINE)
    assert len(commits) == len(emails)
    ret = [[c, e] for c, e in zip(commits, emails)]
    return ret
  
  def getCommitFiles(commit, fileExt=".py"):
    cmd = f"git diff-tree --no-commit-id --name-only -r {commit}"
    files = os.popen(cmd).read().split()
    files = [fname for fname in files if fname.endswith(fileExt)]
    return files
  
  commitInfo = getAllCommit()
  bufferedCommits = readBuffer()
  for pos, [commit, _] in enumerate(commitInfo):
    if commit in bufferedCommits:
      commitInfo[pos].append(bufferedCommits[commit]["python"])
    else:
      commitInfo[pos].append(getCommitFiles(commit))
  
  writeBuffer(commitInfo)
  return commitInfo

def filterAndGetUserFiles(allCommitInfo):
  # We take user in the last commit and filter to get all files from him.
  if allCommitInfo == [] or allCommitInfo[0][1] in USER_WHITE_LIST:
    return []
  
  user = allCommitInfo[0][1]
  files = set()
  for commitInfo in allCommitInfo[: 100]:
    if commitInfo[1] == user:
      files.update(commitInfo[2])
      
  files = [fname for fname in files if os.path.exists(fname)]
  
  return files

if __name__ == "__main__":
  os.system("clear")

  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  #parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
                     #default=False, help="")
  (options, args) = parser.parse_args()
  
  allCommitInfo = getAllCommitInfo()
  files = filterAndGetUserFiles(allCommitInfo)
  error = 0
  for fname in files:
    checker = FormatChecker(fname)
    print("-" * 42, fname)
    error += checker.analyze()
    
  print(f"Total error number: {error}")
  #exit(1)
  exit(error)
