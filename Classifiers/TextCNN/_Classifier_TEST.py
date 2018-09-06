#coding: utf8
#author: Yu Liu (yu.liu55@pactera.com)

from Insight_NLP.Classifiers.TextCNN.Classifier import *
# from Insight_NLP.Classifiers.TextCNN.Data import *
from Insight_NLP.Common import *

if __name__ == '__main__':
  module_path = get_module_path("Insight_NLP.Common")
  data_path = os.path.join(module_path,
                           "Insight_NLP/Classifiers/TextCNN/SampleData")
  
  train_file = os.path.join(data_path, "data.1.train.pydict")
  train_norm_file = train_file.replace(".pydict", ".norm.pydict")
  
  test_file = os.path.join(data_path, "data.1.test.pydict")
  test_norm_file = test_file.replace(".pydict", ".norm.pydict")
  
  #This is only an general example.
  #NOTE, you might define more specific "normalize" function for better
  #performance.
  normalize_data_file(train_file, train_norm_file)
  normalize_data_file(test_file, test_norm_file)
 
  create_vocabulary(train_norm_file, 2, "vob.txt")
  
  trainer = Classifier()
  trainer.train(train_norm_file, test_norm_file,
                vob_file="vob.txt",     # default name
                num_classes=45,
                max_seq_length=64,
                epoch_num=3,
                batch_size=32,
                embedding_size=128,
                kernels="1,1,1,2,3",
                filter_num=128,
                dropout_keep_prob=0.5,
                l2_reg_lambda=0,
                evaluate_frequency=100,
                remove_OOV=False,
                GPU=-1,
                save_model_dir="model")
  
  # predictor = Classifier()
  # predictor.load_model("model")
  
                
  
  
  
  
