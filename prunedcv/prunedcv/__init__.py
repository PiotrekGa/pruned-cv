class ValueTooSmallError(Exception):
    pass


class PrunerCV:

    def __init__(self, n_folds, tolerance):

        if not isinstance(n_folds, int):
            raise TypeError
        if n_folds < 2:
            raise ValueTooSmallError
        if not isinstance(tolerance, float):
            raise TypeError
        if tolerance < 0:
            raise ValueTooSmallError

        self.n_folds = n_folds
        self.tolerance_scaler = tolerance + 1
        self.prun = False
        self.cross_val_score = 0.0
        self.current_folds_list = []
        self.best_folds_list = []
        self.first_run = True

    def _populate_best_folds_list_at_first_run(self, value):

        self.best_folds_list.append(value)

        if len(self.best_folds_list) == self.n_folds:
            self.first_run = False

    def add_fold_value(self, value):

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

    def _decide_prun(self):

        fold_num = len(self.current_folds_list)
        mean_best_folds = sum(self.best_folds_list[:fold_num]) / fold_num
        mean_curr_folds = sum(self.current_folds_list) / fold_num

        if self.n_folds > fold_num >= 2:
            if mean_best_folds * self.tolerance_scaler < mean_curr_folds:
                self.prun = True
                self.cross_val_score = mean_curr_folds
                self.current_folds_list = []

    def _serve_last_fold(self):

        if sum(self.best_folds_list) > sum(self.current_folds_list):
            self.best_folds_list = self.current_folds_list

        self.cross_val_score = sum(self.current_folds_list) / self.n_folds
        self.current_folds_list = []

    def cross_validate_score(self):
        pass
