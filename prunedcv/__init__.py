from sklearn.model_selection import KFold
from sklearn import metrics


class PrunerCV:

    def __init__(self, n_folds, tolerance):

        if not isinstance(n_folds, int):
            raise TypeError
        if n_folds < 2:
            raise ValueError
        if not isinstance(tolerance, float):
            raise TypeError
        if tolerance < 0:
            raise ValueError

        self.n_folds = n_folds
        self.tolerance_scaler = tolerance + 1
        self.prun = False
        self.cross_val_score = 0.0
        self.current_folds_list = []
        self.best_folds_list = []
        self.first_run = True
        self.best_model = None
        self.model = None

    def cross_validate_score(self, model, x, y, metric='mse', shuffle=False, random_state=42):

        self.model = model
        kf = KFold(n_splits=self.n_folds, shuffle=shuffle, random_state=random_state)
        for train_idx, test_idx in kf.split(x, y):
            if not self.prun:
                x_train, y_train = x[train_idx], y[train_idx]
                x_test, y_test = x[test_idx], y[test_idx]
                self.model.fit(x_train, y_train)
                y_test_teor = self.model.predict(x_test)

                if metric == 'mse':
                    self.add_fold_value_and_prun(metrics.mean_squared_error(y_test, y_test_teor))
                if metric == 'mae':
                    self.add_fold_value_and_prun(metrics.mean_absolute_error(y_test, y_test_teor))

        self.prun = False
        return self.cross_val_score

    def add_fold_value_and_prun(self, value):

        if not isinstance(value, float):
            raise TypeError

        if len(self.current_folds_list) == 0:
            self.prun = False

        self.current_folds_list.append(value)

        if self.first_run:
            self._populate_best_folds_list_at_first_run(value)

        self._decide_prun()

        if len(self.current_folds_list) == self.n_folds:
            self._serve_last_fold()

    def _populate_best_folds_list_at_first_run(self, value):

        self.best_folds_list.append(value)

        if len(self.best_folds_list) == self.n_folds:
            self.first_run = False

    def _decide_prun(self):

        fold_num = len(self.current_folds_list)
        mean_best_folds = sum(self.best_folds_list[:fold_num]) / fold_num
        mean_curr_folds = sum(self.current_folds_list) / fold_num

        if self.n_folds > fold_num >= 2:
            if mean_best_folds * self.tolerance_scaler < mean_curr_folds:
                self.prun = True
                self.cross_val_score = mean_curr_folds
                self.current_folds_list = []
                print('trial pruned at {} fold'.format(fold_num))

    def _serve_last_fold(self):

        if sum(self.best_folds_list) > sum(self.current_folds_list):
            self.best_folds_list = self.current_folds_list
            self.best_model = self.model

        self.cross_val_score = sum(self.current_folds_list) / self.n_folds
        self.current_folds_list = []
