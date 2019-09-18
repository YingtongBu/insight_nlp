import abc
import pa_nlp.common as nlp

class Param(abc.ABC):
  def __init__(self, model_name: str):
    self.debug = False
    self.model_name = model_name

    self.path_work = f"_tmp.run.{self.model_name}"
    self.path_model = f"{self.path_work}/model"
    self.init_path_model = f"{self.path_work}/model.init"

    self.lr = 0.001
    self.lr_decay = 0.99
    self.lr_min = 0.0005
    self.epoch_num = 1
    self.batch_size = 32
    self.evaluate_freq = None # in batch number

    self.train_file = ""
    self.eval_files = []

  def verify(self):
    assert not nlp.is_none_or_empty(self.model_name)
    assert not nlp.is_none_or_empty(self.train_file)

    print("-" * 64)
    for key in self.__dict__:
      print(f"{key:20}: {self.__dict__[key]}")
    print("-" * 64)


