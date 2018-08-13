#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from optparse import OptionParser
import re, os

debug = None

class FormatChecker:
  def __init__(self, fname):
    self.basename = os.path.basename(fname)
    self.lines = {} 
    self.lines = [ln.rstrip() for ln in open(fname)]
    self.error = 0
    
  def _ruleAnalyzeAuthor(self):
    for ln in self.lines:
      if "#author:" in ln and "@" in ln:
        return
    print("Error! 文件头上应该有脚本作者以及⼯工作邮箱: "
          "e.g. #author: Tian Xia (summer.xia1@pactera.com)")
    self.error += 1
    
  def _ruleAnalyzeFileLength(self):
    if len(self.lines) > 500:
      print("Error! 单个py文件不能超过500行")
      self.error += 1
      
  def analyze(self):
    self._ruleAnalyzeFileName()
    self._ruleAnalyzeUtf8Setting()
    self._ruleAnalyzeAuthor()
    self._ruleAnalyzeMainFunctionDefinition()
    self._ruleAnalyzeIndentation()
    self._ruleAnalyzeContinuesBlankLines()
    self._ruleAnalyzeConstants()
    self._ruleAnalyzeCompleteWordNaming()
    self._ruleAnlyzeNoneGlobalCodes()
    self._ruleAnalyzeUselessCodes()
    self._ruleAnalyzeFileLength()

    for lnNum, ln in enumerate(self.lines):
      lnNum += 1
      self._ruleAnalyzeMaxLineChars80(lnNum, ln)
      self._ruleAnalyzeTab(lnNum, ln)
      self._ruleAnalyzeClassName(lnNum, ln)
      self._ruleAnalyzeFunctionName(lnNum, ln)
      # self._rule13AnalyzeBlanks(lnNum, ln)
      self._ruleAnalyzeNoBlanksInFuncArguments(lnNum, ln)
      self._ruleAnlyzeCodeReview(lnNum, ln)
      self._ruleAnalyzeChdir(lnNum, ln)
      
    return self.error

  def _ruleAnalyzeFileName(self):
    name = self.basename
    if name[0].islower() or name.isupper() or '_' in name:
      print("Error! 文件名称采用类命名形式，例如FileName.py")
      self.error += 1

  def _ruleAnalyzeUtf8Setting(self):
    for ln in self.lines:
      if "#coding: utf8" in ln or "#coding: utf-8" in ln:
        return
    print("Error! 文件头增加编码设定：#coding: utf8")
    self.error += 1

  def _ruleAnalyzeMainFunctionDefinition(self):
    for ln in self.lines:
      if "__name__" in ln and "__main__" in ln:
        return
    
    print("Error! 每⼀个py⽂件必须添加“伪主函数”部分")
    self.error += 1

  def _ruleAnalyzeChdir(self, lnNum, ln):
    if "chdir(" in ln:
      print(f"Error! line:{lnNum}: 不能使用chdir函数。")
      self.error += 1

  def _ruleAnalyzeTab(self, lnNum, ln):
    if ln == "":
      return
    if "\t" in ln:
      pos = ln.index("\t")
      prefix = ln[: pos].strip()
      if prefix == "":
        print(f"Error! line:{lnNum}: 采⽤用空格代替\\t。")
        self.error += 1

  def _ruleAnalyzeIndentation(self):
    indents = [ln.index(ln.strip()) for ln in self.lines]
    indents = [num for num in indents if num != 0]
    if indents != [] and min(indents) == 4:
      print(f"Error! 每个缩进占2个空格。")
      self.error += 1

  def _ruleAnalyzeMaxLineChars80(self, lnNum, ln):
    length = len(ln)
    if length > 80:
      if debug:
        print(f"line:{lnNum}: length={length}")
      print(f"Error! line:{lnNum} 每⾏行行字符[{length}]不不超过80个。")
      self.error += 1

  def _ruleAnalyzeContinuesBlankLines(self):
    for lnNum in [i for i, x in enumerate(self.lines) if x == ""]:
      if lnNum < len(self.lines) - 1:
        if self.lines[lnNum + 1] == "":
          print(f"Error! line:{lnNum}: 连续空⾏行行不不超过1个。")
          self.error += 1

  def _ruleAnalyzeClassName(self, lnNum, ln):
    if ln.strip().startswith("class "):
      toks = ln.split()
      className = toks[toks.index("class") + 1]
      if className[0].islower() or "_" in className or className.isupper():
        print(f"Error! line:{lnNum} 类命名每个单词首字母大写，中间不加下划线。")
        self.error += 1

  def _ruleAnalyzeFunctionName(self, lnNum, ln):
    if ln.strip().startswith("def "):
      toks = ln.split()
      funcName = toks[toks.index("def") + 1]
      funcName = funcName[: funcName.find("(")]
      if funcName.startswith("__"):
        return
      if funcName.startswith("_"):
        funcName = funcName[1:]
      if funcName[0].isupper() or "_" in funcName or funcName.isupper():
        print(f"Error! line:{lnNum} 函数命名第一个单词小写，后续单词首字母大写。")
        self.error += 1

  def _ruleAnalyzeConstants(self):
    pass

  def _ruleAnalyzeCompleteWordNaming(self):
    pass

  def _ruleAnlyzeNoneGlobalCodes(self):
    pass

  def _ruleAnalyzeBlanks(self, lnNum, ln):
    ln = re.sub("\(.*?\)", "", ln)
    replaced = [
      ("//=", "="),
      ("==", "="),
      ("<=", "="),
      (">=", "="),
      ("!=", "="),
      ("&=", "="),
      ("|=", "="),
      ("^=", "="),
      ("+=", "="),
      ("-=", "="),
      ("*=", "="),
      ("/=", "="),
      ("<=", "="),
    ]
    for s1, s2 in replaced:
      ln = ln.replace(s1, s2)
    if "=" in ln and " = " not in ln and not ln.endswith("="):
      print(f"Error! line:{lnNum} “=”以及其他双目运算符前后各追加⼀个空格。")
      self.error += 1
    if "," in ln and ", " not in ln and not ln.endswith(","):
      print(f"Error! line:{lnNum} 代码的空格，“,”后紧跟⼀个空格。")
      self.error += 1

  def _ruleAnalyzeNoBlanksInFuncArguments(self, lnNum, ln):
    block = " ".join(re.findall("\(.*?\)", ln))
    if " = " in block:
      print(f"Error! line:{lnNum} 函数参数中的=不要两边加空格。")
      self.error += 1

  def _ruleAnalyzeUselessCodes(self):
    pass
  
  def _ruleAnlyzeCodeReview(self, lnNum, ln):
    if "code review" in ln:
      print(f"Error! line:{lnNum} 请及时修改code review的结果。{ln}")
      self.error += 1

def getAllSourceFiles(args):
  def getNextFile():
    if len(args) > 0:
      yield from args
      return

    for root, directories, fileNames in os.walk(os.getcwd()):
      if "Temporary" in root:
        continue
      for fileName in fileNames:
        yield os.path.join(root, fileName)

  for fname in getNextFile():
    if fname.endswith(".py") and "FormatChecker.py" not in fname:
      yield fname

if __name__ == "__main__":
  parser = OptionParser(usage="cmd [file1.py file2.py ...]")
  parser.add_option("--debug", action="store_true")
  #parser.add_option("-q", "--quiet", action = "store_true", dest = "verbose",
                     #default = False, help = "")
  (options, args) = parser.parse_args()

  debug = options.debug
  for fname in getAllSourceFiles(args):
    checker = FormatChecker(fname)
    print("-" * 42, fname)
    checker.analyze()

  print(f"注意：此程序只能分析大多数的格式错误，完整的格式规范请参考文档")
