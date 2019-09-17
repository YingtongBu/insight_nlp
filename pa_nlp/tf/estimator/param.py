import abc
import pa_nlp.common as nlp

class Param(abc.ABC):
  def __init__(self):
    self.model_name = ""

    self.debug = False

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
    self.test_files = []

  def verify(self):
    pass

