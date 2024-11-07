import cv2
import math
import torch
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import xgboost as xgb

from pocket.data import HICODet
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from humfeat import min_clustering
from model import Spatial_Classifier, valid_split, softmax
from greenlab.dft import DiscriminantFeatureTest, FeatureTest
from multiforest_v1 import masking, hoi_dict
import multiforest_v1
import multiforest_XGB

def fit_pred_sum(pred_1, pred_2, obj, eval, save_path):
    from sklearn.metrics import average_precision_score
    pred = np.zeros_like(pred_1)
    weights = np.zeros((80, 117, 2))
    eval_map = []
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if np.sum(eval[obj==i, j]) != 0:
                search_range = [(100-i)*0.01 for i in range(201)]
                ret = []
                for alpha in search_range:
                    beta_range = [(100-i)*0.01 for i in range(201)]
                    for beta in beta_range:
                        if alpha < 0 and beta < 0:
                            ap = 0
                        else:
                            y_score = alpha*pred_1[obj==i, j] + beta*pred_2[obj==i, j]
                            y_score[y_score > 1] = 1
                            # 0.1 for searching
                            y_score[y_score < 0.1] = 0
                            y_true = eval[obj==i, j]
                            ap = average_precision_score(y_true, y_score)
                        ret.append([alpha, beta, ap])
                ret = sorted(ret, key=lambda x: x[2], reverse=True)
                print(f'alpha: {ret[0][0]:.3f}, beta: {ret[0][1]:.3f}; AP: {ret[0][2]:.3f}')
                weights[i, j, :] = np.array([ret[0][0], ret[0][1]])
                pred[obj==i, j] = ret[0][0]*pred_1[obj==i, j] + ret[0][1]*pred_2[obj==i, j]
                eval_map.append(ret[0][-1])
    eval_map = np.array(eval_map)
    print(f'The mAP in evaluation set is: {np.mean(eval_map)}')
    with open(save_path, 'wb') as f:
        np.save(f, weights)
        print(f'The weighted matrix is saved to {save_path}')
    pred[pred > 1] = 1
    pred[pred < 0] = 0
    return weights, pred


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

    test_path = '../hico_20160224_det/images/test2015'
    test_anno = '../hico_20160224_det/instances_test2015.json'
    testset = HICODet(root=test_path, anno_file=test_anno)
    print('HICO-Det dataset loaded')
    
    hoi_list = testset.interactions
    verb_list = testset.verbs
    obj_list = testset.objects

    mask = masking(hoi_list, obj_list, verb_list)
    rel_dict = hoi_dict(hoi_list, obj_list, verb_list)
    datafolder = 'full_feat_nms5'

    print('Testing inference...')
    with open(f'./{datafolder}/x_test_obj_info.npy', 'rb') as f:
        test_obj = np.load(f)
    with open(f'./{datafolder}/y_test.npy', 'rb') as f:
        y_test = np.load(f)

    print(f'============ Starting Ensemble ============')
    
    with open('./two_head/hum_pred_raw.npy', 'rb') as f:
        onehot_hum_pred = np.load(f)

    with open('./two_head/ent_pred_raw.npy', 'rb') as f:
        onehot_ent_pred = np.load(f)

    weight_path = './XGB_head/merging_weights_onehot_v2.npy'
    onehot_weights, onehot_pred = fit_pred_sum(onehot_hum_pred, onehot_ent_pred,
                                               test_obj, y_test, save_path=weight_path)
    
    with open('./XGB_head/hum_pred_th.npy', 'rb') as f:
        hamming_hum_pred = np.load(f)

    with open('./XGB_head/ent_pred_th.npy', 'rb') as f:
        hamming_ent_pred = np.load(f)

    weight_path = './XGB_head/merging_weights_hamming_v2.npy'
    hamming_weights, hamming_pred = fit_pred_sum(hamming_hum_pred, hamming_ent_pred,
                                                 test_obj, y_test, save_path=weight_path)
    weight_path = './XGB_head/merging_weights_hybrid_v2.npy'
    hybrid_weights, hybrid_pred = fit_pred_sum(onehot_pred, hamming_pred,
                                               test_obj, y_test, save_path=weight_path)