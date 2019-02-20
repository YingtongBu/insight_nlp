#coding: utf8
#author: Tian Xia (summer.xia1@pactera.com)

from pa_nlp import *
import pa_nlp.common as nlp

class Measure:
  @staticmethod
  def _WER_single(param):
    ref, hyp = param
    ref_words = ref.split()
    hyp_words = hyp.split()

    d = np.zeros(
      (len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32
    )

    for j in range(len(hyp_words) + 1):
      d[0][j] = j
    for i in range(len(ref_words) + 1):
      d[i][0] = i

    for i in range(1, len(ref_words) + 1):
      for j in range(1, len(hyp_words) + 1):
        if ref_words[i - 1] == hyp_words[j - 1]:
          substitution = d[i - 1][j - 1]
        else:
          substitution = d[i - 1][j - 1] + 1

        insertion = d[i - 1][j] + 1
        deletion = d[i][j - 1] + 1
        d[i][j] = min(substitution, insertion, deletion)

    return d[len(ref_words)][len(hyp_words)], len(ref_words)

  @staticmethod
  def calc_WER(ref_list: list, hyp_list: list,
               parallel: bool=False, case_sensitive: bool=False):
    '''
    In the parallel mode, the multiprocess.Pool() would leads memory leak.
    '''
    assert type(ref_list) is list and type(ref_list[0]) is str
    if not case_sensitive:
      ref_list = [ref.lower() for ref in ref_list]
      hyp_list = [hyp.lower() for hyp in hyp_list]

    if parallel:
      pool = mp.Pool()
      error_list, len_list = nlp.split_to_sublist(
        pool.map(Measure._WER_single, zip(ref_list, hyp_list))
      )
      pool.close()

    else:
      error_list, len_list = nlp.split_to_sublist(
        [Measure._WER_single([ref, hyp])
         for ref, hyp in zip(ref_list, hyp_list)]
      )

    error = sum(error_list)
    ref_count = max(1, sum(len_list))

    return error / ref_count

  @staticmethod
  def calc_precision_recall_fvalue(true_labels: list, preded_labels: list):
    '''
    :return (recall, precision, f) for each label, and
    (average_f, weighted_f, precision) for all labels.
    '''
    assert len(true_labels) == len(preded_labels)
    true_label_num = defaultdict(int)
    pred_label_num = defaultdict(int)
    correct_label = defaultdict(int)
    
    for t_label, p_label in zip(true_labels, preded_labels):
      true_label_num[t_label] += 1
      pred_label_num[p_label] += 1
      if t_label == p_label:
        correct_label[t_label] += 1
        
    result = dict()
    label_stat = Counter(true_labels)
    for label in label_stat.keys():
      correct = correct_label.get(label, 0)
      recall = correct / (true_label_num.get(label, 0) + nlp.EPSILON)
      prec = correct / (pred_label_num.get(label, 0) + nlp.EPSILON)
      f_value = 2 * (recall * prec) / (recall + prec + nlp.EPSILON)
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

  @staticmethod
  def calc_intervals_accurarcy(true_labels_list: list, pred_labels_list: list):
    '''
    :param true_labels_list: [[(0., 2.0), (3.4, 4.5)], [(2. 0, 4.0)]]
    :param pred_labels_list:  [[(0., 2.0), (3.4, 4.5)], [(2. 0, 4.0)]]
    :return: accuracy, recall, f-value, more information.
    '''
    assert type(true_labels_list) == type(true_labels_list[0]) == list
    assert type(pred_labels_list) == type(pred_labels_list[0]) == list

    results = [
      Measure._intervals_accurarcy_single(true_labels, pred_labels)
      for true_labels, pred_labels in zip(true_labels_list, pred_labels_list)
    ]
    correct = sum([r["correct"] for r in results])
    true_label_num = sum([r["true_label_num"] for r in results])
    pred_label_num = sum([r["pred_label_num"] for r in results])

    recall = correct / true_label_num
    accuracy = correct /pred_label_num
    f = 2 * recall * accuracy / (recall + accuracy + nlp.EPSILON)

    return {
      "recall": round(recall, 4),
      "accuracy": round(accuracy, 4),
      "f": round(f, 4),
      "details:": results
    }

  @staticmethod
  def _intervals_accurarcy_single(true_labels: list, pred_labels: list):
    matched_label_num = np.zeros([len(pred_labels)], np.int)
    missing_labels = []
    correct_num = 0
    for label in true_labels:
      for idx, pred_label in enumerate(pred_labels):
        if nlp.segment_contain(pred_label, label):
          matched_label_num[idx] += 1
          correct_num += 1
          break
      else:
        missing_labels.append(label)

    wrong_indices = np.where(matched_label_num == 0)[0]
    wrong_labels = []
    for index in wrong_indices:
      wrong_labels.append(pred_labels[index])

    total_pred_num = len(wrong_labels) + sum(matched_label_num)

    return {
      "correct": correct_num,
      "true_label_num": len(true_labels),
      "pred_label_num": total_pred_num,
      "missing": missing_labels,
      "wrong": wrong_labels,
    }