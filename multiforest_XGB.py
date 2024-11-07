import cv2
import math
import torch
import dill as pickle
#import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import xgboost as xgb

from joblib import Parallel, delayed
from pocket.data import HICODet
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score
from sklearn.metrics import recall_score
from humfeat import min_clustering
from model import Spatial_Classifier, valid_split, softmax
from greenlab.dft import DiscriminantFeatureTest, FeatureTest
from dimXGB import focal_loss, multidimXGB

def masking(hoi_list, obj_list, verb_list):
    ''' Binary masking to eliminating the impossible object and relation pair
    Arg(s):
        hoi_list(list)   : the str list of hoi pairs (verb + ' ' + obj)
        obj_list(list)   : the str list of objects
        verb_list(list)  : the str list of verbs
    Return(s):
        masking(np.array): the binary array with size (obj, verb)
    '''
    masking = np.zeros((len(obj_list), len(verb_list)))
    for hoi in hoi_list:
        verb, obj = hoi.split()
        masking[obj_list.index(obj), verb_list.index(verb)] = 1
    return masking

def hoi_dict(hoi_list, obj_list, verb_list):
    ''' Query the hoi index by the (verb, obj) pair
    Arg(s):
        hoi_list(list)   : the str list of hoi pairs (verb + ' ' + obj)
        obj_list(list)   : the str list of objects
        verb_list(list)  : the str list of verbs
    Return(s):
        hoi_dict(dict)   : the dict with key (vb, obj) and value (hoi)
    '''
    hoi_dict = {}
    for idx, hoi in enumerate(hoi_list):
        verb, obj = hoi.split()
        hoi_dict[(verb_list.index(verb), obj_list.index(obj))] = idx
    return hoi_dict

class JointlySaab:
    def __init__(self, kernel_size=5):
        self.feature_length = kernel_size * kernel_size * 256

    def fit(self, train_x: np.array, train_y: np.array):
        if len(np.unique(train_y)) == 1:
            self.weights = 0.5 * np.ones((self.feature_length, 2))
        else:
            from sklearn.linear_model import LogisticRegression
            self.weights = np.zeros((self.feature_length, 2))
            x_1 = train_x[:, :self.feature_length]
            x_2 = train_x[:, self.feature_length:self.feature_length*2]
            x = np.stack([x_1, x_2], axis=1)
            for i in range(self.feature_length):
                clf = LogisticRegression(fit_intercept=False)
                clf.fit(x[:, :, i], train_y)
                self.weights[i] = clf.coef_

    def transform(self, test_x: np.array):
        x_1 = test_x[:, :self.feature_length]
        x_2 = test_x[:, self.feature_length:self.feature_length*2]
        x = x_1*self.weights[:, 0] + x_2*self.weights[:, 1]
        test_x = np.concatenate([x, test_x[:, self.feature_length*2:]], axis=1)
        #print(f'The transformed feature with shape: {test_x.shape}')
        return test_x

def Sample_Weights(train_label):
    labels, counts =  np.unique(train_label, return_counts=True)
    weights = dict(zip(labels, np.sqrt(np.max(counts) / counts)))
    print(weights)
    return np.array([ weights[label] for label in train_label ])

class forest_cls:
    def __init__(self, masking, obj_list, n_neighbors=1, merge=False):
        self.cls = {}
        self.dft = {}
        self.masking = masking
        self.obj_list = obj_list
        self.threshold = None
        self.conditional = None
        self.sim = None
        self.num_feat = 1000
        self.n_neighbors = n_neighbors
        self.merge = merge
        print(f'The number of features: {self.num_feat}')
        if merge:
            print(f'Using the transformed features.')
        
    def load_config(self, 
                    temp = {
                            'objective':'binary:logistic',
                            'gamma':1.2,
                            'learning_rate':0.005,
                            'reg_lambda':1.0,
                            'subsample':0.9,
                            'colsample_bytree':0.8,
                            'eval_metric':['error', 'map', 'logloss'],
                            'verbosity':0,
                            'n_estimators':1000,
                            'min_child_weight':1.2
                            },
                    use_gpu=False):
        for i in range(self.masking.shape[0]):
            #'max_depth':6
            self.cls[i] = []
            config = copy.deepcopy(temp)
            config['n_estimators'] = 1000
            self.dft[i] = []
            for _ in range(int(sum(self.masking[i, :]))):
                config = copy.deepcopy(temp)
                if use_gpu:
                    config['tree_method'] = 'gpu_hist'
                self.cls[i].append(multidimXGB())

    def grouping(self, y, th=None):
        from sklearn.cluster import KMeans
        prob = np.divide(y, np.sqrt(np.sum(y, axis=0))+1e-9)
        kmeans = KMeans(n_clusters=2, n_init='auto').fit(prob.T)
        ent_groups = kmeans.labels_
        print(f'The Kmeans clustering result: {ent_groups}')
        major, minor = [], []
        for index, label in enumerate(ent_groups):
            if label == 0:
                major.append(index)
            elif label == 1:
                minor.append(index)
        print('The major classes are:')
        print(major)
        print('The minor classes are:')
        print(minor)
        return major, minor
    
    def confidence_score(self, idx: int, train_x: np.array, stage: str='surp'):
        # Building confidence from previous stage
        if stage == 'surp':
            dft = self.dft_surp[idx]
            bst = self.surp[idx]
            th = self.partition[idx]['surp_th']
        elif stage == 'major':
            dft = self.dft_major[idx]
            bst = self.major[idx]
            th = self.partition[idx]['major_th']
        elif stage == 'minor':
            dft = self.dft_minor[idx]
            bst = self.minor[idx]
            th = self.partition[idx]['minor_th']
        if self.merge:
            x_feat = dft['saab'].transform(train_x)
            x_feat = dft['dft'].transform(x_feat, n_selected=self.num_feat)
        else:
            x_feat = dft['dft'].transform(train_x, n_selected=self.num_feat)
        logits = bst.predict(xgb.DMatrix(x_feat))
        confidence = np.exp(logits) / (1+np.exp(logits))
        return confidence, th

    def train(self, x: np.array, entity: np.array, y: np.array,
              savedir='./forest_model/', eval_set=None,
              use_extra=False, resume=0):

        for idx in range(self.masking.shape[0])[resume:]:
            print('')
            print(f'Training...')
            print(f'Entity {idx}: {self.obj_list[idx]}')
            train_x, train_y, label = self.prepare_data(x, y, entity, idx)
            no_interaction_id = label.index(57)
            print(f'No interaction id: {no_interaction_id}')

            for rel in range(train_y.shape[1]):
                
                sample_y = train_y
                sample_x = train_x
                
                extra_data = use_extra or (np.sum(train_y[:, rel]) == 0)
                if extra_data:
                    print('Using more entities...')
                    extra_y = y[y[:, label[rel]]==1, :][:, label]
                    extra_x = x[y[:, label[rel]]==1, :]
                    sample_y = np.concatenate([train_y, extra_y])
                    sample_x = np.concatenate([train_x, extra_x])

                if eval_set is None:
                    x_train, x_valid, y_train, y_valid = valid_split(sample_x, sample_y, train_ratio=0.8)
                else:
                    x_train, y_train = sample_x, sample_y
                    x_valid, valid_y, _ = self.prepare_data(eval_set[0], eval_set[2], eval_set[1], idx)
                    y_valid = valid_y[:, rel]
                    #x_valid = x_valid[~np.all(valid_y==0, axis=1), :]
                    #y_valid = valid_y[~np.all(valid_y==0, axis=1), rel]

                print(f'Training Data with shape: {x_train.shape}, {y_train.shape}')
                print(f'Validation with shape: {x_valid.shape}, {valid_y.shape}')

                XGB_cls = self.cls[idx][rel]
                XGB_cls.train(x_train, y_train, rel)
                XGB_cls.validation(x_valid, valid_y)

                #import pdb; pdb.set_trace()
                y_pred = XGB_cls.predict(x_valid)
                print(f'The validation mAP: {average_precision_score(y_valid, y_pred):.4f}')
                del x_train, x_valid, y_train, y_valid, sample_x, sample_y

                print(f'The classifier is saved to {savedir}{idx}_{rel}.pkl')
                with open(f'{savedir}{idx}_{rel}.pkl', 'wb') as f:
                    pickle.dump(XGB_cls, f)
                del XGB_cls
        '''
        print(f'Save partition to: {savedir}partition.pkl')
        with open(f'{savedir}partition.pkl', 'wb') as f:
            pickle.dump(self.partition, f)
        if train_log is not None:
            print(log)
            with open(f'{savedir}/{train_log}', 'wb') as f:
                pickle.dump(log, f)
        '''

    def prepare_data(self, x, y, entity, idx):
        label = np.where(self.masking[idx]!=0)[0].tolist()
        print(label)
        train_label = y[entity==idx][:, label]
        train_x = x[entity==idx]
        return train_x, train_label, label

    def load(self, model_dir='./forest_model/'):
        print(f'Load Classifiers from: {model_dir}')
        for idx in range(self.masking.shape[0]):
            for rel in range(int(sum(self.masking[idx, :]))):
                #self.cls[idx][rel].load(f'{model_dir}{idx}_{rel}.json')
                #print(f'Load Classifier {idx}_{rel}')
                with open(f'{model_dir}{idx}_{rel}.pkl', 'rb') as f:
                    #self.cls[idx].append(pickle.load(f))
                    self.cls[idx][rel] = pickle.load(f)

    def testbystage(self, x, entity, y, th=0.5):
        ret = []
        for idx in range(self.masking.shape[0]):
            print(f'Testing...')
            print(f'Entity {idx}: {self.obj_list[idx]}')
            x_test, y_test = self.prepare_data(x, y, entity, idx)
            y_pred = []
            for rel in range(y_test.shape[1]):
                if self.dft[idx][rel] != '':
                    #x_feat = self.dft[idx][rel].select(x_test, self.num_feat)
                    x_feat = self.dft[idx][rel].transform(x_test, self.num_feat)
                    test_dmatrix = xgb.DMatrix(x_feat)
                    tmp = self.cls[idx][rel].predict(test_dmatrix).reshape(-1, 1)
                    tmp = np.exp(tmp) / (1+np.exp(tmp))
                    y_pred.append(tmp)
                else:
                    y_pred.append(np.zeros((x_test.shape[0], 1)))
            y_pred = np.concatenate(y_pred, axis=1)
            print(f'The shape of ground truth: {y_test.shape}')
            print(f'The shape of predictions : {y_pred.shape}')
            y_pred[y_pred >= th] = 1
            y_pred[y_pred < th] = 0
            self.cls[idx][0].show_statistic(y_test.astype(int), y_pred.astype(int))
            ret.append(accuracy_score(y_test.astype(int), y_pred.astype(int)))
        return ret

    def coco2hico(self, labels):
        ''' Converse the coco instnace label to hico labels
        Arg(s):
            label(list): The COCO index
        Return(s):
            lable(list): The HICO index
        '''
        conversion = [
             4, 47, 24, 46, 34, 35, 21, 59, 13,  1, 14,  8, 73, 39, 45, 50,  5,
            55,  2, 51, 15, 67, 56, 74, 57, 19, 41, 60, 16, 54, 20, 10, 42, 29,
            23, 78, 26, 17, 52, 66, 33, 43, 63, 68,  3, 64, 49, 69, 12,  0, 53,
            58, 72, 65, 48, 76, 18, 71, 36, 30, 31, 44, 32, 11, 28, 37, 77, 38,
            27, 70, 61, 79,  9,  6,  7, 62, 25, 75, 40, 22
        ]
        return [conversion.index(label) for label in labels]

    def predictbyent(self, feat, ent):
        parallel = False
        label = np.where(self.masking[ent]!=0)[0]
        probs = np.zeros(117)
        feat = feat.reshape((1, -1))
        def wrap_func(rel):
            preds = self.cls[ent][rel].predict(feat)
            return preds[0]
        if parallel:
            #import pdb; pdb.set_trace()
            probs[label] = np.array(Parallel(n_jobs=3)(delayed(wrap_func)(rel) for rel in range(len(self.cls[ent]))))
            #print(probs[label])
        else:
            for rel in range(len(self.cls[ent])):
                preds = self.cls[ent][rel].predict(feat)
                probs[label[rel]] = preds[0]
        return probs

    def predict_prob(self, x, entity, y=None):
        ''' binary classifiers for each triplet
        '''
        ret = []
        for feat, ent in zip(x, entity):

            if self.conditional is None:
                probs = self.predictbyent(feat, ent)
            else:
                probs = np.zeros(117)
                weights = 0
                for neighbor, similarity in self.sim[ent]:
                    #probs += self.predictbyent(feat, neighbor) * similarity
                    probs = np.maximum(probs, self.predictbyent(feat, neighbor)*similarity)
                    weights += similarity
                #probs = probs / weights
                probs = probs * self.masking[ent]

            probs = self.predictbyent(feat, ent)
            if not (self.threshold is None):
                for i in range(117):
                    th = self.threshold[ent][i]
                    if probs[i] >= 1.5*th:
                        probs[i] = 1
                    elif probs[i] >= th and probs[i] < 1.5*th:
                        probs[i] = probs[i]/(1.5*th)
                    else:
                        probs[i] = 0.4*(probs[i]/th)
                        #probs[label[rel]] = preds[0]

            ret.append(probs)

        ret = np.array(ret)
        
        if not (y is None):
            stat = np.zeros_like(ret)
            if not (self.threshold is None):
                stat[ret >= 0.5] = 1
                stat[ret < 0.5] = 0
                self.cls[0][0].show_statistic(y, stat)
            else:
                stat[ret >= 0.5] = 1
                stat[ret < 0.5] = 0
                self.cls[0][0].show_statistic(y, stat)
        return ret

    def thresholding(self, x, entity, y, save_path='./forest_model/threshold.npy'):
        
        ret = self.predict_prob(x, entity)

        with open('./predict_prob.npy', 'wb') as f:
            np.save(f, ret)
        pred = np.zeros_like(ret)
        pred[ret < 0.5] = 0
        pred[ret >= 0.5] = 1
        self.cls[0][0].show_statistic(y, pred)

        search_range = [i*0.01 for i in range(99)]

        def compute_th(x, y, search_space):
            ret = []
            if (np.sum(x) == 0) and (np.sum(y) == 0):
                return 1
            from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
            from sklearn.metrics import average_precision_score
            for th in search_space:
                tmp = np.zeros_like(x)
                tmp[x>th] = x[x>th]/(1.5*th+1e-9)
                tmp[x<=th] = x[x<=th]*0.4/(th+1e-9)
                tmp[tmp>1] = 1
                #ret.append((th, f1_score(y, tmp)))
                ret.append((th, average_precision_score(y, tmp)))
            ret = sorted(ret, key=lambda x: x[1], reverse=True)
            #print(ret[:3])
            if ret[0][0] != 0:
                return ret[0][0]
            else:
                return 1e-4

        th = np.zeros_like(self.masking)
        for i in range(th.shape[0]):
            label = np.where(entity == i)[0].tolist()
            for j in range(th.shape[1]):
                pred = ret[label, j]
                gt = y[label, j]
                max = np.max(pred)
                min = np.min(pred)
                search_range = [min+(max-min)*(i*0.01) for i in range(100)]
                th[i, j] = compute_th(pred, gt, search_range)
                pred[pred > th[i, j]] = 1
                pred[pred <= th[i, j]] = 0
                ret[label, j] = pred

        self.threshold = th
        print(th.shape)
        with open(save_path, 'wb') as f:
            np.save(f, th)
            print(f'The thresholding matrix is saved to {save_path}')
    
    def load_threshold(self, path='./forest_model/threshold.npy'):
        print(f'The thresholding matrix is loaded from:\n{path}')
        with open(path, 'rb') as f:
            self.threshold = np.load(f)

    def load_freq(self, path='./forest_model/frequency.npy'):
        print(f'The conditional matrix is loaded from:\n{path}')
        with open(path, 'rb') as f:
            self.conditional = np.load(f)
        self.conditional[:, 57] = 0
        self.sim = []
        n_neighbors = self.n_neighbors
        print(f'The weighted sum of {n_neighbors} nearest neighbors.')
        for i in range(80):
            tmp = []
            for j in range(80):
                num = self.conditional[i]@self.conditional[j].T
                dom = (self.conditional[i]@self.conditional[i].T) * (self.conditional[j]@self.conditional[j].T)
                sim = num/np.sqrt(dom)
                tmp.append((j, sim))
            tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
            #print(tmp[:n_neighbors])
            #import pdb; pdb.set_trace()
            self.sim.append(tmp[:n_neighbors])


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

    forest = forest_cls(mask, obj_list)
    forest.load_config(use_gpu=False)

    print('Loading features...')
    with open('./full_feat/x_train_sp_roi3.npy', 'rb') as f:
        train_sp = np.load(f)
    print('Train the classifiers with the pretrained ResNet50 background features')
    bgfeat_path = './full_feat/x_train_bg_roi3.npy'
    with open(bgfeat_path, 'rb') as f:
        train_bg = np.load(f)
    humfeat_path = './full_feat/x_train_human_roi3.npy'
    with open(humfeat_path, 'rb') as f:
        train_hum = np.load(f)
        print(f'loading humfeat from: {humfeat_path}')
    with open('./full_feat/x_train_obj_roi3.npy', 'rb') as f:
        train_ent = np.load(f)
    with open('./full_feat/x_train_obj_info.npy', 'rb') as f:
        train_obj = np.load(f)
    with open('./full_feat/y_train.npy', 'rb') as f:
        y = np.load(f)
    
    #x = np.concatenate([train_hum, train_bg, train_sp], axis=1)
    x = np.concatenate([train_hum, train_ent, train_bg, train_sp], axis=1)

    print('Testing inference...')
    with open('./full_feat/x_test_sp_roi3.npy', 'rb') as f:
        test_sp = np.load(f)
    with open('./full_feat/x_test_bg_roi3.npy', 'rb') as f:
        test_bg = np.load(f)
    with open('./full_feat/x_test_human_roi3.npy', 'rb') as f:
        test_hum = np.load(f)
    with open('./full_feat/x_test_obj_roi3.npy', 'rb') as f:
        test_ent = np.load(f)
    with open('./full_feat/x_test_obj_info.npy', 'rb') as f:
        test_obj = np.load(f)
    with open('./full_feat/y_test.npy', 'rb') as f:
        y_test = np.load(f)
    #x_test = np.concatenate([test_hum, test_sp], axis=1)
    #x_test = np.concatenate([test_hum, test_bg, test_sp], axis=1)
    x_test = np.concatenate([test_hum, test_ent, test_bg, test_sp], axis=1)
    
    nestmodel_path = './neg_forest/full_roi3_focal/'
    print(f'The nest model is saved to {nestmodel_path}')
    forest.train(x, train_obj, y, savedir=nestmodel_path, eval_set=[x_test, test_obj, y_test])
    del x, y
    forest.load_config(use_gpu=False)
    forest.load(model_dir=nestmodel_path)
    forest.load_freq(path='./forest_model/frequency.npy')

    #tree.load(model_dir='./nest_model/rel_cluster/')
    print(f'============ Starting Inference ============')
    import time;
    start_time = time.time()
    path = './neg_forest/threshold_focal_f1.npy'
    forest.thresholding(x_test, test_obj, y_test, save_path=path)
    forest.predict_prob(x_test, test_obj, y=y_test)
    end_time = time.time()
    print(f'=============== End Inference ===============')
    print(f'Time Consuming: {end_time - start_time:.3f} sec')
