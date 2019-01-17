#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

import common as nlp
from collections import defaultdict, Counter
from common import EPSILON

class Measure:
  @staticmethod
  def calc_precision_recall_fvalue(true_labels, preded_labels):
    '''
    :return (recall, precision, f) for each label, and
    (average_f, weighted_f, precision) for all labels.
    '''
    assert len(true_labels) == len(preded_labels)
    true_label_num = defaultdict(int)
    preded_label_num = defaultdict(int)
    correct_label = defaultdict(int)
    
    for t_label, p_label in zip(true_labels, preded_labels):
      true_label_num[t_label] += 1
      preded_label_num[p_label] += 1
      if t_label == p_label:
        correct_label[t_label] += 1
        
    result = dict()
    label_stat = Counter(true_labels)
    for label in label_stat.keys():
      correct = correct_label.get(label, 0)
      recall = correct / (true_label_num.get(label, 0) + EPSILON)
      prec = correct / (preded_label_num.get(label, 0) + EPSILON)
      f_value = 2 * (recall * prec) / (recall + prec + EPSILON)
      result[label] = {
        "recall": round(recall, 4),
        "precision": round(prec, 4),
        "f": round(f_value, 4),
      }
     
    total_f = sum([result[label]["f"] for label in label_stat.keys()])
    avg_f_value = total_f / len(label_stat)
    result["average_f"] = round(avg_f_value, 4)
    
    total_f = sum([result[label]["f"] * label_stat.get(label, 0)
                   for label in label_stat.keys()])
    weighted_f_value = total_f / len(true_labels)
    result["weighted_f"] = round(weighted_f_value, 4)
    
    result["precision"] = round(sum(correct_label.values()) / len(true_labels),
                                4)
    
    return result

