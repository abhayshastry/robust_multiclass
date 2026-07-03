# robust_multiclass
# Creates an object which can be called during classificaiton
import numpy as np
from itertools import combinations
import math_utils as mu  #providing primal_robust_socp

class RobustOVOModel:
    """
    Multiclass Robust SVM via One-vs-One with precomputed kernels.
    After .fit(), you can call .predict(K_vt) on any test kernel (n_test x n_train).
    """
    def __init__(self, classes, pairs, pair_params):
        self.classes_ = classes              # shape (K,)
        self.pairs_ = pairs                  # list of tuples (ci, cj)
        self.pair_params_ = pair_params      # list of dicts: {"idx_pair", "beta", "b"}

        # map class label; index 0,1,...,K-1
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}

    def decision_function_pairwise(self, K_vt):
        """
        Return pairwise decision values in OvO order (n_samples, n_pairs),
        just like sklearn's SVC(ovo).decision_function.
        """
        n_samples = K_vt.shape[0]
        scores = np.empty((n_samples, len(self.pair_params_)), dtype=float)

        for k, p in enumerate(self.pair_params_):
            cols = p["idx_pair"]                  # indices into training set (columns)
            beta, b = p["beta"], p["b"]
            scores[:, k] = K_vt[:, cols] @ beta + b
        return scores

    def predict(self, K_vt):
        """
        Majority vote with margin tie-break, mirroring libsvm/sklearn OvO.
        K_vt must be (n_test, n_train) with columns aligned to the training order.
        """
        n_samples = K_vt.shape[0]
        K = len(self.classes_)
        votes   = np.zeros((n_samples, K), dtype=int)
        margins = np.zeros((n_samples, K), dtype=float)

        for (ci, cj), p in zip(self.pairs_, self.pair_params_):
            i = self._class_to_idx[ci]
            j = self._class_to_idx[cj]
            cols = p["idx_pair"]
            beta, b = p["beta"], p["b"]

            s = K_vt[:, cols] @ beta + b    # (n_samples,)

            votes[s >= 0, i] += 1
            votes[s <  0, j] += 1
            margins[:, i] += np.clip(s,  0, None)
            margins[:, j] += np.clip(-s, 0, None)

        y_pred = np.empty(n_samples, dtype=self.classes_.dtype)
        for r in range(n_samples):
            best = votes[r].max()
            tied = np.flatnonzero(votes[r] == best)
            y_pred[r] = self.classes_[tied[0]] if len(tied) == 1 else self.classes_[tied[np.argmax(margins[r, tied])]]
        return y_pred

#fit function to get training params for all pairs of classes. Exactly keeping up the sklearn style.
def fit_robust_ovo(K_train, y_train, *, C=1.0, delta_1=0.01, delta_2=0.01, shots=np.inf, conf_int = 0):
    """
    Train once on the training kernel. Returns a RobustOVOModel that can
    predict on any test kernel K_vt (n_test x n_train) later.
    """
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    idx_by_class = {c: np.flatnonzero(y_train == c) for c in classes}

    pairs = []
    pair_params = []

    for ci, cj in combinations(classes, 2):
        idx_i = idx_by_class[ci]
        idx_j = idx_by_class[cj]
        idx_pair = np.concatenate([idx_i, idx_j])

        # binary labels for this pair
        y_bin = np.concatenate([np.ones(len(idx_i)), -np.ones(len(idx_j))])

        # training block for this pair
        K_tt = K_train[np.ix_(idx_pair, idx_pair)]
        K_tt = np.clip(K_tt, 0, 1)
        assert(mu.isPSD(K_tt))
        assert np.all(K_tt >= 0), f"Found values < 0: {K_tt.min()}"
        assert np.all(K_tt <= 1), f"Found values > 1: {K_tt.max()}"
        assert not np.isnan(K_tt).any(), "Found NaN values"

        # robust binary fit on the pair
        if conf_int == 0 or shots == np.inf:
            beta, b, *_ = mu.primal_robust_socp(K_tt, y_bin, C=C, delta_1=delta_1, delta_2=delta_2, shots=shots)
        else:
            if shots!=np.inf:
                beta, b, *_ = mu.primal_robust_l2(K_tt, y_bin, C=C, delta_1 = delta_1, delta_2 = delta_2, shots = shots, ci = conf_int)
        pairs.append((ci, cj))
        pair_params.append({"idx_pair": idx_pair, "beta": beta, "b": b})

    return RobustOVOModel(classes, pairs, pair_params)



