"""v6 model wrappers (shared by training and the live watcher so pickles load in both)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import feat6

SEED = 0


def gbm(reg=False):
    if reg:      # stronger regularisation: small trees, big leaves, half the features per split (indicators are redundant)
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=600,
                                              l2_regularization=3.0, max_features=0.5, early_stopping=False, random_state=SEED)
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=False, random_state=SEED)


class LR:
    """Standardised logistic regression (NaN -> training median)."""
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y):
        X = X[self.cols]; self.med = X.median(); Z = X.fillna(self.med)
        self.mu = Z.mean(); self.sd = Z.std().replace(0, 1)
        self.m = LogisticRegression(C=0.5, max_iter=5000).fit(((Z - self.mu) / self.sd).to_numpy(), y); return self
    def predict_proba(self, X):
        Z = X[self.cols].fillna(self.med); return self.m.predict_proba(((Z - self.mu) / self.sd).to_numpy())


class G:
    def __init__(self, cols, reg=False): self.cols = cols; self.reg = reg
    def fit(self, X, y): self.m = gbm(self.reg).fit(X[self.cols].to_numpy(np.float32), y); return self
    def predict_proba(self, X): return self.m.predict_proba(X[self.cols].to_numpy(np.float32))


class Ens:
    """average of a logistic and a regularised GBM on the same features"""
    def __init__(self, cols): self.cols = cols; self.a = LR(cols); self.b = G(cols, reg=True)
    def fit(self, X, y): self.a.fit(X, y); self.b.fit(X, y); return self
    def predict_proba(self, X): return (self.a.predict_proba(X) + self.b.predict_proba(X)) / 2


def make(spec, sets):
    kind, fs = spec
    cols = feat6.V5 if fs == "v5" else sets[fs]
    return {"lr": LR, "gbm": G, "gbmr": lambda c: G(c, reg=True), "ens": Ens}[kind](cols)


