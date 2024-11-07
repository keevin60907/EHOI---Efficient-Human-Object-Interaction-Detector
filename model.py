import cv2
import math
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from PIL import Image, ImageDraw
from pocket.data import HICODet
from torchvision.ops.boxes import batched_nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from labeling import spatial_feat
from humfeat import min_clustering

rel_cluster = [9, 2, 8, 1, 0, 6, 6, 4, 9, 8,
               6, 9, 1, 9, 4, 1, 6, 9, 9, 8,
               1, 0, 7, 4, 6, 2, 6, 1, 5, 1,
               2, 6, 5, 7, 6, 8, 3, 3, 0, 7,
               9, 2, 9, 5, 8, 7, 6, 2, 6, 6,
               6, 1, 0, 9, 4, 6, 9, 0, 9, 6,
               9, 9, 3, 9, 1, 7, 1, 4, 6, 1,
               2, 3, 3, 1, 6, 3, 2, 2, 7, 2,
               7, 8, 9, 6, 8, 1, 6, 3, 4, 1,
               8, 1, 6, 5, 9, 6, 6, 9, 3, 6,
               6, 1, 6, 1, 8, 2, 1, 7, 3, 6,
               3, 0, 6, 9, 5, 6, 9]

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
    
def valid_split(x, y, train_ratio):
    labels = list(set(y))
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x, y = x[idx, :], y[idx]
    x_train, y_train = [], []
    x_valid, y_valid = [], []
    for label in labels:
        idx = np.where(y == label)[0]
        num_data = idx.shape[0]
        x_train.append(x[idx[:math.ceil(num_data*train_ratio)]])
        y_train.append(y[idx[:math.ceil(num_data*train_ratio)]])
        x_valid.append(x[idx[math.floor(num_data*train_ratio):]])
        y_valid.append(y[idx[math.floor(num_data*train_ratio):]])
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    shuffle = np.arange(x_train.shape[0])
    np.random.shuffle(shuffle)
    x_train, y_train = x_train[shuffle, :], y_train[shuffle]

    shuffle = np.arange(x_valid.shape[0])
    np.random.shuffle(shuffle)
    x_valid, y_valid = x_valid[shuffle, :], y_valid[shuffle]

    return x_train, x_valid, y_train, y_valid

def random_split(x, y, train_ratio):
    num_data = x.shape[0]
    idx = np.arange(num_data)
    np.random.shuffle(idx)
    x_train = x[idx[:math.floor(num_data*train_ratio)], :]
    x_valid = x[idx[math.floor(num_data*train_ratio):], :]
    y_train = y[idx[:math.floor(num_data*train_ratio)]]
    y_valid = y[idx[math.floor(num_data*train_ratio):]]
    return x_train, x_valid, y_train, y_valid

def batch_feat(input, model, batch=100):
    ''' Batch input to get the Saab features'''    
    n_batch = input.shape[0] // batch + 1
    feat = []
    for i in range(n_batch):
        if i == (n_batch - 1):
            batch_input = input[i*batch:]
        else:
            batch_input = input[i*batch:(i+1)*batch]
        feat.append(model(batch_input))
    return np.concatenate(feat, axis=0)

def focal_loss(pred, dtrain):
        def robust_pow(num_base, num_pow):
        # numpy does not permit negative numbers to fractional power
        # use this to perform the power algorithmic
            return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)
        
        gamma_indct = 5.0
        # retrieve data from dtrain matrix
        label = dtrain.get_label()
        # compute the prediction with sigmoid
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        # gradient
        # complex gradient with different parts
        g1 = sigmoid_pred * (1 - sigmoid_pred)
        g2 = label + ((-1) ** label) * sigmoid_pred
        g3 = sigmoid_pred + label - 1
        g4 = 1 - label - ((-1) ** label) * sigmoid_pred
        g5 = label + ((-1) ** label) * sigmoid_pred
        # combine the gradient
        grad = gamma_indct * g3 * robust_pow(g2, gamma_indct) * np.log(g4 + 1e-9) + \
               ((-1) ** label) * robust_pow(g5, (gamma_indct + 1))
        # combine the gradient parts to get hessian components
        hess_1 = robust_pow(g2, gamma_indct) + \
                 gamma_indct * ((-1) ** label) * g3 * robust_pow(g2, (gamma_indct - 1))
        hess_2 = ((-1) ** label) * g3 * robust_pow(g2, gamma_indct) / g4
        # get the final 2nd order derivative
        hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct +
                (gamma_indct + 1) * robust_pow(g5, gamma_indct)) * g1

        return grad, hess

class Spatial_Classifier:
    def __init__(self, config):
        self.config = config
        self.COCO = [
            'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 
            'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
            'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
            'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle', 
            'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear',
            'hair_drier', 'toothbrush' ]
        
    def train(self, dmat_train, dmat_valid, search=False):
        self.dmat_train, self.dmat_valid = dmat_train, dmat_valid
        watch_list = [(self.dmat_train, 'train'), (self.dmat_valid, 'valid')]
        self.num_boost_round = 10000
        self.early_stopping_rounds = 20
        self.decay_rate = 0.95
        self.evals_result = {}
        self.scheduler = xgb.callback.LearningRateScheduler(lambda epoch:max(0.01, 0.2*self.decay_rate**epoch))
        if not search:
            #self.clf = xgb.train(self.config, self.dmat_train, self.num_boost_round,
            #        evals=watch_list, early_stopping_rounds=self.early_stopping_rounds,
            #        evals_result=self.evals_result, verbose_eval=True, callbacks=[self.scheduler])
            self.clf = xgb.train(self.config, self.dmat_train, self.num_boost_round,
                    obj=focal_loss,
                    evals=watch_list, early_stopping_rounds=self.early_stopping_rounds,
                    evals_result=self.evals_result, verbose_eval=True)
        else:
            param_grid = {
                'max_depth': [3, 4, 5], 
                'n_estimators':[100, 500, 1000]
                }
            from sklearn.model_selection import GridSearchCV, StratifiedKFold
            self.clf = xgb.XGBClassifier(**self.config,
                    callbacks=[self.scheduler])
            skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 1001)
            grid_search = GridSearchCV(
                estimator = self.clf,
                param_grid = param_grid,
                cv = skf,
                scoring = 'roc_auc',
                verbose = True
            )
            fit_params={'early_stopping_rounds':self.early_stopping_rounds,
                    'eval_set' : [[dmat_train, dmat_valid]]}
            grid_search.fit(dmat_train, dmat_valid, **fit_params)

            self.clf = grid_search.best_estimator_
    
    def save(self, filepath):
        self.clf.save_model(filepath)
        #with open(filepath.replace('.json', '.pkl'), 'wb') as f:
        #    pickle.dump(self.evals_result, f)

    def load(self, filepath):
        self.clf = xgb.Booster(self.config)
        self.clf.load_model(filepath)
        #with open(filepath.replace('.json', '.pkl'), 'rb') as f:
        #    self.evals_result = pickle.load(f)

    def predict(self, x_test):
        iteration_range = (0, self.clf.best_iteration+1)
        y_pred = self.clf.predict(x_test, iteration_range)
        return y_pred
    
    def predict_prob(self, x_test):
        y_pred = self.clf.predict_proba(x_test)
        return y_pred

    def plot_history(self, fig_folder):
        # preparing evaluation metric plots
        results = self.evals_result
        epochs = len(results['valid']['mlogloss'])
        x_axis = range(0, epochs)

        # xgboost 'mlogloss' plot
        fig, ax = plt.subplots(figsize=(9,5))
        ax.plot(x_axis, results['train']['mlogloss'], label='Train')
        ax.plot(x_axis, results['valid']['mlogloss'], label='Test')
        ax.legend()
        plt.ylabel('mlogloss')
        plt.title('GridSearchCV XGBoost mlogloss')
        plt.savefig(f'{fig_folder}/mlogloss.jpg')

        # xgboost 'merror' plot
        fig, ax = plt.subplots(figsize=(9,5))
        ax.plot(x_axis, results['train']['merror'], label='Train')
        ax.plot(x_axis, results['valid']['merror'], label='Test')
        ax.legend()
        plt.ylabel('merror')
        plt.title('GridSearchCV XGBoost merror')
        plt.savefig(f'{fig_folder}/merror.jpg')

    def show_statistic(self, y_test, y_pred):

        print('')
        print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
        #print(f'Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.2f}')

        print('')
        print(f'Micro Precision: {precision_score(y_test, y_pred, average="micro"):.2f}')
        print(f'Micro Recall: {recall_score(y_test, y_pred, average="micro"):.2f}')
        print(f'Micro F1-score: {f1_score(y_test, y_pred, average="micro"):.2f}')

        print('')
        print(f'Macro Precision: {precision_score(y_test, y_pred, average="macro"):.2f}')
        print(f'Macro Recall: {recall_score(y_test, y_pred, average="macro"):.2f}')
        print(f'Macro F1-score: {f1_score(y_test, y_pred, average="macro"):.2f}')

        print('\n--------------- Classification Report ---------------\n')
        print(classification_report(y_test, y_pred))
        print('---------------------- XGBoost ----------------------')

    def plot_importance(self, file_name):
        fig, ax = plt.subplots(figsize=(9,50))
        plot_importance(self.clf, ax=ax)
        plt.savefig(file_name)

    def topk_imoprtance(self, topk=15):
        importance = list(self.clf.get_fscore().items())
        ret = sorted(importance, key=lambda x:x[1], reverse=True)
        return ret[:topk]

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

if __name__ == '__main__':
    '''
    with open('test.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)

    metric = MeanAveragePrecision(class_metrics=False)
    # The tutorial is in:
    # https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
    metric.update(preds, target)
    https://stackoverflow.com/questions/66785587/how-do-i-use-validation-sets-on-multioutputregressor-for-xgbregressor
    '''
    print('Loading features...')
    with open('./spatial_feat/x_train.npy', 'rb') as f:
        x = np.load(f)
    #with open('./spatial_feat/x_train_human_img.npy', 'rb') as f:
    #    train_hum = np.load(f).transpose(0, 3, 1, 2)
    with open('./spatial_feat/x_train_pixel2hop_c500_clustersample.npy', 'rb') as f:
        train_hum = np.load(f)
    with open('./spatial_feat/x_train_obj.npy', 'rb') as f:
        train_obj = np.load(f)
    with open('./spatial_feat/y_train.npy', 'rb') as f:
        y = np.load(f)

    with open('./spatial_feat/x_test.npy', 'rb') as f:
        x_test = np.load(f)
    #with open('./spatial_feat/x_test_human_img.npy', 'rb') as f:
    #    test_hum = np.load(f).transpose(0, 3, 1, 2)
    with open('./spatial_feat/x_test_pixel2hop_c500_clustersample.npy', 'rb') as f:
        test_hum = np.load(f)
    with open('./spatial_feat/x_test_obj.npy', 'rb') as f:
        test_obj = np.load(f)
    with open('./spatial_feat/y_test.npy', 'rb') as f:
        y_test = np.load(f)
    
    #model_path = './XGB_models/spatial_model/spatialfeat_label117.json'
    # If you are performing multi-label classification, please don't specify the objective function 
    # (or specify it as binary:logistic, which is done inside XGBoost).
    # 'objective':'multi:softprob', 
    mapping, _ = min_clustering(y)
    grouping = True
    if grouping:
        #train_labels = np.array([mapping[i][0] for i in y]) # first level
        #test_labels = np.array([mapping[i][0] for i in y_test])
        train_labels = np.array([rel_cluster[i] for i in y]) # first level
        test_labels = np.array([rel_cluster[i] for i in y_test])
    else:
        train_labels = y
        test_labels = y_test
    num_class = len(list(set(train_labels)))
    print(f'Number of clsses: {num_class}')

    config = {
        #'tree_method': 'gpu_hist',
        'objective':'multi:softprob',
        'num_class':num_class,
        'gamma':0,
        'learning_rate':0.2,
        'max_depth':6,
        'reg_lambda':1,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'eval_metric':['merror','mlogloss'],
        'seed':42,
    }
    sp_clf = Spatial_Classifier(config)
    embedding = True
    if embedding:
        print('Using fastText word embedding with dimension 300')
        with open('./spatial_feat/fastText.npy', 'rb') as f:
            object_info = np.load(f)
            print('Performing SVD for (80, 300) embeddings...')
            _, _, d = np.linalg.svd(object_info, full_matrices=False)
    else:
        print('Using one-hot encoding with dimension 80')
        object_info = np.identity(80)

    x = np.concatenate([x, train_hum, object_info[train_obj, :]@d.T], axis=1)
    print(x.shape)
    x_train, x_valid, y_train, y_valid = random_split(x, train_labels, train_ratio=0.8)
    train_dmatrix = xgb.DMatrix(x_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(x_valid, label=y_valid)
    sp_clf.train(train_dmatrix, valid_dmatrix)

    clf_savepath = './XGB_models/hop2_emb_cluster_sample/hum_emb_cluster_sample.json'
    sp_clf.save(clf_savepath)
    print(f'XGBoost classifier is saved to the path: \n{clf_savepath}')
    #sp_clf.load('./XGB_models/spatial_entity/spatial_entity_label117.json')
    
    fig_savepath = './figs/XGB_hop2_emb_cluster_sample'
    sp_clf.plot_history(fig_savepath)
    print(f'Corresponding figures are saved to the path: \n{fig_savepath}')
    print(sp_clf.topk_imoprtance())

    x_test = np.concatenate([x_test, test_hum, object_info[test_obj, :]@d.T], axis=1)
    test_dmatrix = xgb.DMatrix(x_test)
    y_pred = sp_clf.predict(test_dmatrix)
    y_pred = softmax(y_pred, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    sp_clf.show_statistic(test_labels, y_pred)

    sp_clf.plot_importance(f'{fig_savepath}/importance.jpg')
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(test_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f'{fig_savepath}/confusion.jpg')

    '''
    sp_clf.plot_importance('b.jpg')

    le = LabelEncoder()
    enc_y_train = le.fit_transform(y)
    enc_y_test = le.fit_transform(y_test)

    y_prob = sp_clf.predict_prob(x_test)
    enc_idx = [ i for i in range(y_prob.shape[1]) ]
    idx = le.inverse_transform(enc_idx)
    y_prob = y_prob[:, idx]

    enc_y_pred = sp_clf.predict(x_test)
    y_pred = le.inverse_transform(enc_y_pred)
    sp_clf.show_statistic(y_test, y_pred)
    '''
