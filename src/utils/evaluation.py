import os
import numpy as np

def cal_AP(query_label, result_top_k):
    result = np.array([res.split(os.path.sep)[-2] for res in result_top_k])
    if np.sum(result == query_label) == 0:
        return 0
    else:
        AP = np.sum((np.arange(np.sum(result == query_label)) + 1) / (np.where(result == query_label)[0] + 1)) / np.sum(result == query_label)
        return AP  