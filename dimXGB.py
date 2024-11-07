import numpy as np
from typing import Union, Tuple
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score

import dill as pickle
from joblib import Parallel, delayed
# Parallel(n_jobs=2)(delayed(func)(i ** 2) for i in range(10))

import xgboost as xgb
from greenlab.dft import FeatureTest

class node:
    '''
    Internal data structure while computing Huffman coding
    '''
    def __init__(self, symbol, count, lchild=None, rchild=None):
        self.symbol = symbol
        self.count = count
        self.lchild = lchild
        self.rchild = rchild
    def __repr__(self):
        return f'{self.symbol}: {self.count}'

def Huffman(freq):
    '''
    Core Huffman Algorithm
    '''
    while (len(freq) > 1):
        freq = sorted(freq, key=lambda x: x.count)
        merge = node(f'[{freq[0].symbol},{freq[1].symbol}]',
                    freq[0].count+freq[1].count,
                    lchild=freq[0], rchild=freq[1])
        freq = freq[2:]
        freq.append(merge)
    ret = {}

    def traverse(node, code, ret):
        '''
        DFS traversal helper function
        '''
        if node.lchild is not None:
            traverse(node.lchild, code+'0', ret)
        if node.rchild is not None:
            traverse(node.rchild, code+'1', ret)
        if node.lchild is None and node.rchild is None:
            ret[node.symbol] = code
    traverse(freq[0], '', ret)

    return ret

def partition(y: np.array) -> dict:
    '''
    Arg(s):
        y(np.array)     : one-hot encoding for the classification data
    Return(s):
        codebook(dict)  : codebook with symbols and padded length codewords
    '''
    codebook = {}
    freq = []
    n_class = y.shape[1]
    freq.append(node('null', np.sum(np.all(y==0, axis=1))))
    for i in range(n_class):
        freq.append(node(i, np.sum(y[:, i])))
    codebook = Huffman(freq)
    print(f'Codebook without padding:')
    print(codebook)
    return codebook

def focal_loss(y_true, y_pred, gamma=2.0):
    def robust_pow(num_base, num_pow):
    # numpy does not permit negative numbers to fractional power
    # use this to perform the power algorithmic
        return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)
    
    gamma_indct = gamma
    # retrieve data from dtrain matrix
    label = y_true
    # compute the prediction with sigmoid
    sigmoid_pred = 1.0 / (1.0 + np.exp(-y_pred))
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

class multidimXGB:

    def __init__(self, num_feat=1000, n_jobs=1):
        self.num_feat = num_feat
        self.n_jobs = n_jobs
        self.config = {
                        'gamma':1.3,
                        'learning_rate':0.01,
                        'reg_lambda':1.0,
                        'subsample':0.8,
                        'colsample_bytree':0.7,
                        'eval_metric':['error', 'map', 'logloss'],
                        'verbosity':0,
                        'n_estimators':300,
                        'min_child_weight':1,
                        'early_stopping_rounds':10
                        }
        self.decay_rate = 0.95
        self.scheduler = xgb.callback.LearningRateScheduler(lambda epoch:max(0.01, 0.2*self.decay_rate**epoch))
        self.Hamming = {
                        'null': '1111111',  0: '0000000',  1: '0001111',  2:'0010011',
                             3: '0011100',  4: '0100101',  5: '0101010',  6:'0110110',
                             7: '0111001',  8: '1000110',  9: '1001001', 10:'1010101',
                            11: '1011010', 12: '1100011', 13: '1101100', 14:'1110000'
                        }
        self.Hamming15 = {
                        'null': '111110111',  0: '000000000',  1: '000011001',  2:'000101110',
                             3: '000110111',  4: '001000110',  5: '001011111',  6:'001101000',
                             7: '001110001',  8: '010001010',  9: '010010011', 10:'010100100',
                            11: '010111101', 12: '011001100', 13: '011010101', 14:'011100010',
                            15: '011111011', 16: '100001100', 17: '100010101', 18:'100100010',
        }

    def train(self, x: np.array, y: np.array, label: int, device: str='cuda'):
        #self.codebook = partition(y)
        n_class = y.shape[1]
        self.codebook = {}
        if n_class <= 15:
            self.codebook['null'] = self.Hamming['null']
            for i in range(n_class):
                self.codebook[i] = self.Hamming[i]
        else:
            self.codebook['null'] = self.Hamming15['null']
            for i in range(n_class):
                self.codebook[i] = self.Hamming15[i]
        self.codebook = self.rm_redundant_bit(self.codebook)
        print(f'Hamming codebook: {self.codebook}')
        self.label = label
        self.dft, self.clf, self.merge = self.multidimXGB(x, y, self.codebook, label, device)
        self.n_stage = len(self.dft)


    def rm_redundant_bit(self, codebook: dict):
        codes = list(codebook.values())
        bitID, partition = [], []
        for i in range(len(codes[0])):
            bitplane = ''.join([code[i] for code in codes])
            if bitplane not in partition:
                bitID.append(i)
                partition.append(bitplane)
        if len(bitID) == range(len(codes[0])):
            return codebook
        else:
            new_codebook = {}
            for key, values in codebook.items():
                new_codebook[key] = ''.join([values[i] for i in bitID])
            return new_codebook

    def multidimXGB(self, x: np.array, y: np.array,
                    codebook: dict, label: int,
                    device: str='cuda'):
        #maxlength = len(sorted(codebook.values(), key=lambda x: len(x))[-1])
        #classifiers = [[] for _ in range(y.shape[1])]
        classifiers, dfts = [], []
        codeword = codebook[label]
        n_hyperplane = len(codebook[label])
        print(f'Target dimension: {label} with {n_hyperplane} partitions.')
        def parallel_train(hyperplane):
            x_train, y_train = self.construct_data(x, y, codebook, label, hyperplane)
            dft = FeatureTest()
            dft.fit(x_train, y_train, n_bins=16)
            x_train = dft.transform(x_train, n_selected=self.num_feat)

            print('Using Weighted Samples')
            from sklearn.utils import class_weight
            classes_weights = class_weight.compute_sample_weight(
                                class_weight='balanced',
                                y=y_train
                                )
            print(f'Shape of the smaple weights: {classes_weights.shape}')
            print(f'The weights of positive samples: {np.unique(classes_weights)}')

            clf = xgb.XGBClassifier(**self.config,
                                    callbacks=[self.scheduler],
                                    objective=focal_loss,
                                    device=device)
            #clf.set_params()
            clf.fit(x_train, y_train,
                    sample_weight=classes_weights,
                    eval_set=[(x_train, y_train)])
            print(f'The binary accuracy score for {hyperplane}th hyperplane: {clf.score(x_train, y_train)}')
            return dft, clf
        returns = Parallel(n_jobs=self.n_jobs)(delayed(parallel_train)(hyperplane) for hyperplane in range(n_hyperplane))
        dfts = [item[0] for item in returns]
        classifiers = [item[1] for item in returns]

        if n_hyperplane > 1:
            print(f'Train the merge step')
            x_train = []
            y_train = y[:, label]
            for hyperplane in range(n_hyperplane):
                code = int(codeword[hyperplane])
                x_feat = dfts[hyperplane].transform(x, n_selected=self.num_feat)
                if classifiers[hyperplane].classes_.shape[0] > 1:
                    #x_train.append(classifiers[hyperplane].predict_proba(x_feat)[:, code])
                    x_train.append(classifiers[hyperplane].predict_proba(x_feat)[:, 1])
                else:
                    x_train.append(np.ones_like(y_train))
            x_train = np.stack(x_train, axis=1)
            try:
                merge_clf = LinearDiscriminantAnalysis()
                merge_clf.fit(x_train, y_train)
            except:
                print('The features are linearly dependent')
                if np.unique(y_train).shape[0] == 1:
                    print('Training data only contains one class')
                    merge_clf = RandomForestClassifier()
                    merge_clf.fit(x_train, y_train)
                else:
                    merge_clf = SVC(probability=True)
                    merge_clf.fit(x_train, y_train)
            print(f'The merge process accuracy: {merge_clf.score(x_train, y_train)}')
        else:
            merge_clf = None
        return dfts, classifiers, merge_clf

    def validation(self, x: np.array, y: np.array):
        print(f'Target dimension: {self.label} with {self.n_stage} partitions.')
        codeword = self.codebook[self.label]
        x_prob = []
        def wrap_pred(hyperplane):
            #for hyperplane in range(self.n_stage):
            code = int(codeword[hyperplane])
            x_valid, y_valid = self.construct_data(x, y, self.codebook, self.label, hyperplane)
            dft = self.dft[hyperplane]
            x_valid = dft.transform(x_valid, n_selected=self.num_feat)
            clf = self.clf[hyperplane]
            print(f'The validation accuracy score for {hyperplane}th hyperplane: {clf.score(x_valid, y_valid)}')
            x_feat = dft.transform(x, n_selected=self.num_feat)
            if clf.classes_.shape[0] > 1:
                #x_prob.append(clf.predict_proba(x_feat)[:, code])
                #x_prob.append(clf.predict_proba(x_feat)[:, 1])
                return clf.predict_proba(x_feat)[:, 1]
            else:
                #x_prob.append(np.ones_like(y[:, self.label]))
                return np.ones_like(y[:, self.label])
        x_prob = Parallel(n_jobs=self.n_jobs)(delayed(wrap_pred)(hyperplane) for hyperplane in range(self.n_stage))

        if self.n_stage > 1:
            x_prob = np.stack(x_prob, axis=1)
            print(f'The validation accuracy score: {self.merge.score(x_prob, y[:, self.label])}')
        #else:
        #    print(f'The validation accuracy score: {clf.score(x_valid, y_valid)}')

    def construct_data(self, x: np.array, y: np.array,
                   codebook: dict, label: Union[int, str],
                   hyperplane: int) -> Tuple[np.array, np.array]:
        codeword = codebook[label]
        partition_0, partition_1 = [], []
        for idx, code in codebook.items():
            if len(code) <= hyperplane:
                partition_0.append(idx)
            elif codeword[hyperplane] != code[hyperplane]:
                partition_0.append(idx)
            else:
                partition_1.append(idx)

        partition_x, partition_y = [], []
        
        for idx in partition_0:
            if idx == 'null':
                partition_x.append(x[np.all(y==0, axis=1), :])
                partition_y.append(np.zeros(np.sum(np.all(y==0, axis=1))))
            else:
                partition_x.append(x[y[:, idx]==1, :])
                partition_y.append(np.zeros(np.sum(y[:, idx]==1)))
        for idx in partition_1:
            if idx == 'null':
                partition_x.append(x[np.all(y==0, axis=1), :])
                partition_y.append(np.ones(np.sum(np.all(y==0, axis=1))))
            else:
                partition_x.append(x[y[:, idx]==1, :])
                partition_y.append(np.ones(np.sum(y[:, idx]==1)))
        partition_x = np.concatenate(partition_x, axis=0)
        partition_y = np.concatenate(partition_y, axis=0)
        #if int(codeword[hyperplane]) == 0:
        #    partition_y = 1 - partition_y
        return partition_x, partition_y

    def predict(self, x_test: np.array):
        self.n_jobs = 1
        prob = np.ones(x_test.shape[0])
        codeword = self.codebook[self.label]
        x_prob = []
        #for i in range(self.n_stage):
        def wrap_pred(hyperplane):
            #code = int(codeword[hyperplane])
            x_feat = self.dft[hyperplane].transform(x_test, n_selected=self.num_feat)
            if self.clf[hyperplane].classes_.shape[0] > 1:
                #x_prob.append(self.clf[i].predict_proba(x_feat)[:, code])
                #x_prob.append(self.clf[hyperplane].predict_proba(x_feat)[:, 1])
                return self.clf[hyperplane].predict_proba(x_feat)[:, 1]
                #prob = prob * self.clf[i].predict_proba(x_feat)[:, code]
            else:
                #x_prob.append(prob)
                return prob
        x_prob = Parallel(n_jobs=self.n_jobs)(delayed(wrap_pred)(hyperplane) for hyperplane in range(self.n_stage))

        if self.n_stage > 1:
            x_prob = np.stack(x_prob, axis=1)
            if self.merge.classes_.shape[0] > 1:
                prob = self.merge.predict_proba(x_prob)
                #print(prob[:, 1])
                return prob[:, 1]
            else:
                prob = self.merge.predict(x_prob)
                return prob
        else:
            return x_prob[0]
    
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

if __name__ == '__main__':

    n_data = 1000
    n_class = 7
    n_feature = 3000
    x = np.random.rand(n_data, n_feature)
    y = np.random.randint(0,2,(n_data,n_class))
    #codebook = partition(y)
    LDA_cls = multidimXGB()
    import time
    start = time.time()
    LDA_cls.train(x, y, 0, device='cpu')
    y_pred = LDA_cls.predict(x)
    end = time.time()
    print(f'CPU time: {end-start} sec')
    print(y_pred[:10])
    start = time.time()
    LDA_cls.train(x, y, 0, device='cuda')
    y_pred = LDA_cls.predict(x)
    end = time.time()
    print(f'GPU time: {end-start} sec')
    print(y_pred[:10])

    with open(f'XGB.pkl', 'wb') as f:
        pickle.dump(LDA_cls, f)
    with open(f'XGB.pkl', 'rb') as f:
        load_model = pickle.load(f)
    y_pred_0 = load_model.predict(x)
    print(y_pred_0[:10])

    from sklearn.metrics import accuracy_score
    ret = []
    for i in range(n_class):
        LDA_cls = multidimXGB()
        LDA_cls.train(x, y, i, device='cuda')
        ret.append(LDA_cls.predict(x))
    y_pred = np.stack(ret, axis=1)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    print(accuracy_score(y, y_pred))