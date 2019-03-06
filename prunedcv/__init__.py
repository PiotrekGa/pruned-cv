from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np


class PrunerCV:

    def __init__(self, n_splits, tolerance, splits_to_start_pruning=2):

        if not isinstance(n_splits, int):
            raise TypeError
        if n_splits < 2:
            raise ValueError
        if not isinstance(tolerance, float):
            raise TypeError
        if tolerance < 0:
            raise ValueError

        self.n_splits = n_splits
        self.tolerance = tolerance
        self.tolerance_scaler = tolerance + 1
        self.splits_to_start_pruning = splits_to_start_pruning
        self.prun = False
        self.cross_val_score = 0.0
        self.current_splits_list = []
        self.best_splits_list = []
        self.first_run = True
        self.best_model = None
        self.model = None

    def cross_validate_score(self, model, x, y, metric='mse', shuffle=False, random_state=42):

        if not isinstance(x, np.ndarray):
            raise TypeError

        if not isinstance(y, np.ndarray):
            raise TypeError

        self.model = model
        kf = KFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state)
        for train_idx, test_idx in kf.split(x, y):
            if not self.prun:
                x_train, y_train = x[train_idx], y[train_idx]
                x_test, y_test = x[test_idx], y[test_idx]
                self.model.fit(x_train, y_train)
                y_test_teor = self.model.predict(x_test)

                if metric == 'mse':
                    self.add_split_value_and_prun(metrics.mean_squared_error(y_test, y_test_teor))
                if metric == 'mae':
                    self.add_split_value_and_prun(metrics.mean_absolute_error(y_test, y_test_teor))

        self.prun = False
        return self.cross_val_score

    def add_split_value_and_prun(self, value):

        if not isinstance(value, float):
            raise TypeError

        if len(self.current_splits_list) == 0:
            self.prun = False

        self.current_splits_list.append(value)

        if self.first_run:
            self._populate_best_splits_list_at_first_run(value)

        self._decide_prun()

        if len(self.current_splits_list) == self.n_splits:
            self._serve_last_split()

    def _populate_best_splits_list_at_first_run(self, value):

        self.best_splits_list.append(value)

        if len(self.best_splits_list) == self.n_splits:
            self.first_run = False

    def _decide_prun(self):

        split_num = len(self.current_splits_list)
        mean_best_splits = sum(self.best_splits_list[:split_num]) / split_num
        mean_curr_splits = sum(self.current_splits_list) / split_num

        if self.n_splits > split_num >= self.splits_to_start_pruning:
            if mean_best_splits * self.tolerance_scaler < mean_curr_splits:
                self.prun = True
                self.cross_val_score = mean_curr_splits
                self.current_splits_list = []
                print('trial pruned at {} fold'.format(split_num))

    def _serve_last_split(self):

        if sum(self.best_splits_list) > sum(self.current_splits_list):
            self.best_splits_list = self.current_splits_list
            self.best_model = self.model

        self.cross_val_score = sum(self.current_splits_list) / self.n_splits
        self.current_splits_list = []
