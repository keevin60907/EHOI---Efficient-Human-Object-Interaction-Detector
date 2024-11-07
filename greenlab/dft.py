import numpy as np
import multiprocessing as mp
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelBinarizer


class DiscriminantFeatureTest:
    def __init__(self, max_depth=1, n_jobs=-1):
        self.max_depth = max_depth
        if n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs

    def fit(self, X, y):
        if self.n_jobs == 1:
            self.loss = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                tree = DecisionTreeClassifier(max_depth=self.max_depth).fit(
                    X[:, i : i + 1], y
                )
                prob = tree.predict_proba(X[:, i : i + 1])
                self.loss[i] = log_loss(y, prob)
        else:
            with mp.Pool(self.n_jobs) as pool:
                self.loss = np.array(
                    pool.map(
                        self._fit, [(X[:, i : i + 1], y) for i in range(X.shape[1])]
                    )
                )
        self.rank = np.argsort(self.loss)
        self.sorted_loss = self.loss[self.rank]
        return self

    def _fit(self, args):
        X, y = args
        tree = DecisionTreeClassifier(max_depth=self.max_depth).fit(X, y)
        prob = tree.predict_proba(X)
        return log_loss(y, prob)

    def select(self, X, n):
        return X[:, self.rank[:n]]

    def plot_loss(self, sorted=True, path=None):
        if sorted:
            plt.plot(self.sorted_loss)
        else:
            plt.plot(self.loss)
        plt.xlabel("Feature Index")
        plt.ylabel("Loss")
        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

class Conditional_DFT(DiscriminantFeatureTest):
    def __init__(self, extra_info, max_depth=1, n_jobs=1):
        super().__init__(max_depth=1, n_jobs=1)
        self.extra_info = extra_info

    def fit(self, X, y):
        if self.n_jobs == 1:
            self.loss = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                tree = DecisionTreeClassifier(max_depth=self.max_depth).fit(
                    np.concatenate([X[:, i : i + 1], self.extra_info], axis=1), 
                    y
                )
                prob = tree.predict_proba(np.concatenate([X[:, i : i + 1], self.extra_info], axis=1))
                self.loss[i] = log_loss(y, prob)
        else:
            with mp.Pool(self.n_jobs) as pool:
                self.loss = np.array(
                    pool.map(
                        self._fit, [(X[:, i : i + 1], y) for i in range(X.shape[1])]
                    )
                )
        self.rank = np.argsort(self.loss)
        self.sorted_loss = self.loss[self.rank]
        return self

    def _fit(self, args):
        X, y = args
        X = np.concatenate([X, self.extra_info], axis=1)
        tree = DecisionTreeClassifier(max_depth=self.max_depth).fit(X, y)
        prob = tree.predict_proba(X)
        return log_loss(y, prob)

    def select(self, X, n):
        return X[:, self.rank[:n]]

    def plot_loss(self, sorted=True, path=None):
        if sorted:
            plt.plot(self.sorted_loss)
        else:
            plt.plot(self.loss)
        plt.xlabel("Feature Index")
        plt.ylabel("Loss")
        if path is not None:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

class FeatureTest:
    def __init__(self, loss='bce'):
        assert loss in ['bce', 'ce', 'rmse', 'focal'],\
              f'loss not supported. Please select from ["bce", "ce", "rmse", "focal"].'
        self.loss = loss
        self.dim_loss = dict()
        self.sorted_features = None
        self.dim = 0
        self.validation_fold = 5
        self.n_candidates = 1000

    def fit(self, X, y, n_bins, outliers=False):

        from sklearn.model_selection import StratifiedKFold
        self.dim = X.shape[1]
        kfolds = StratifiedKFold(n_splits=self.validation_fold)
        folds_feature = []
        try:
            for train, _ in kfolds.split(X, y):
                dim_loss = dict()
                for d in range(self.dim):
                    min_partition_loss = self.get_min_partition_loss(X[train, d], y[train], n_bins, outliers)
                    dim_loss[d] = min_partition_loss
                dim_loss = {k: v for k, v in sorted(dim_loss.items(), key=lambda item: item[1])}
                sorted_features = np.array(list(dim_loss.keys()))[:self.n_candidates]
                folds_feature.append(sorted_features)
            folds_feature = np.array(folds_feature)
            feature, counts = np.unique(folds_feature, return_counts=True)
            self.feature_idx = []
            for feat, cnt in zip(feature, counts):
                if cnt > (self.validation_fold / 2) + 1:
                    self.feature_idx.append(feat)
            print(f'The number of chosen dimension: {len(self.feature_idx)}')
            print(f'The top 20 influential feature index:{self.feature_idx[:20]}')
            self.sorted_features = self.feature_idx
        except:
            dim_loss = dict()
            for d in range(self.dim):
                    min_partition_loss = self.get_min_partition_loss(X[:, d], y, n_bins, outliers)
                    dim_loss[d] = min_partition_loss
            dim_loss = {k: v for k, v in sorted(dim_loss.items(), key=lambda item: item[1])}
            self.feature_idx = np.array(list(dim_loss.keys()))[:self.n_candidates]
            print(f'The number of chosen dimension: {len(self.feature_idx)}')
            print(f'The top 20 influential feature index:{self.feature_idx[:20]}')
            self.sorted_features = self.feature_idx

    def transform(self, X, n_selected=None):
        assert self.sorted_features is not None, f'Run fit() before selecting features.'
        assert X.shape[1] == self.dim, f'Expect feature dimension {self.dim}, but got {X.shape[1]}.'
        #return X[:, self.sorted_features[np.arange(n_selected)]]
        return X[:, self.feature_idx]

    def fit_transform(self, X, y, n_bins, n_selected):
        self.fit(X, y, n_bins)
        return self.transform(X, n_selected)

    def get_min_partition_loss(self, f_1d, y, n_bins, outliers=False):
        if outliers:
            f_1d, y = self.remove_outliers(f_1d, y)
        min_partition_loss = float('inf')
        f_min, f_max = f_1d.min(), f_1d.max()
        bin_width = (f_max - f_min) / n_bins
        for i in range(1, n_bins):
            partition_point = f_min + i * bin_width
            y_l, y_r = y[f_1d <= partition_point], y[f_1d > partition_point]
            partition_loss = self.get_loss(y_l, y_r)
            if partition_loss < min_partition_loss:
                min_partition_loss = partition_loss
        return min_partition_loss

    def get_loss(self, y_l, y_r):
        n1, n2 = len(y_l), len(y_r)
        if self.loss == 'bce':
            if n1 == 0:
                lp = 0
            else:
                lp = y_l.mean()
            if lp == 1 or lp == 0:
                lh = 0.0
            else:
                lh = np.sum(-y_l * np.log2(lp) - (1 - y_l) * np.log2(1 - lp))
            if n2 == 0:
                rp = 0
            else:
                rp = y_r.mean()
            if rp == 1 or rp == 0:
                rh = 0.0
            else:
                rh = np.sum(-y_r * np.log2(rp) - (1 - y_r) * np.log2(1 - rp))
            return (lh + rh) / (n1 + n2)
        elif self.loss == 'ce':
            llb = MyLabelBinarizer()
            y_l = llb.fit_transform(y_l)
            lp = y_l.mean(axis=0)
            lh = np.sum(-y_l * np.log2(lp))
            rlb = MyLabelBinarizer()
            y_r = rlb.fit_transform(y_r)
            rp = y_r.mean(axis=0)
            rh = np.sum(-y_r * np.log2(rp))
            return (lh + rh) / (n1 + n2)
        elif self.loss == 'rmse':
            left_mse = ((y_l - y_l.mean()) ** 2).sum()
            right_mse = ((y_r - y_r.mean()) ** 2).sum()
            return np.sqrt((left_mse + right_mse) / (n1 + n2))
        elif self.loss == 'focal':
            gamma = 2.0
            if n1 == 0:
                lp = 0
            else:
                lp = y_l.mean()
            if lp == 1 or lp == 0:
                lh = 0.0
            else:
                #lh = np.sum(-y_l * np.log2(lp) - (1 - y_l) * np.log2(1 - lp))
                lh = np.sum(- y_l * ((1-lp) ** gamma) * np.log2(lp) \
                            - (1 - y_l) * ((lp) ** gamma) * np.log2(1 - lp))
            if n2 == 0:
                rp = 0
            else:
                rp = y_r.mean()
            if rp == 1 or rp == 0:
                rh = 0.0
            else:
                #rh = np.sum(-y_r * np.log2(rp) - (1 - y_r) * np.log2(1 - rp))
                rh = np.sum(- y_r * ((1-rp) ** gamma) * np.log2(rp) \
                            - (1 - y_r) * ((rp) ** gamma) * np.log2(1 - rp))
            return (lh + rh) / (n1 + n2)
        else:
            pass

    @staticmethod
    def remove_outliers(f_1d, y, n_std=2.0):
        """Remove outliers for the regression problem."""
        f_mean, f_std = f_1d.mean(), f_1d.std()
        return f_1d[np.abs(f_1d - f_mean) <= n_std * f_std], y[np.abs(f_1d - f_mean) <= n_std * f_std]

class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if len(self.classes_) == 1:
            return 1 - Y
        elif len(self.classes_) == 2:
            return np.hstack((Y, 1-Y))
        else:
            return Y

if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100000, n_features=1000, n_informative=2, n_classes=2
    )
    info = np.random.rand(100000, 50)
    #dft = DiscriminantFeatureTest(max_depth=1)
    dft = Conditional_DFT(info, max_depth=1, n_jobs=1)
    dft.fit(X, y)
    dft.plot_loss(path='./a.jpg')
    dft.plot_loss(path='./b.jpg', sorted=False)
    '''
    dft = DiscriminantFeatureTest(max_depth=3)
    dft.fit(X, y)
    dft.plot_loss()
    dft.plot_loss(sorted=False)
    '''