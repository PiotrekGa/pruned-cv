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
        self.value_to_return = 0.0
        self.current_folds_list = []
        self.best_folds_list = []
        self.first_run = True

    def _populate_best_folds_at_first_run(self, value):

        if self.first_run:
            self.best_folds_list.append(value)

            if len(self.best_folds_list) == self.n_folds:
                self.first_run = False

    def add_folds_value(self, value):

        if not isinstance(value, float):
            raise TypeError

        if len(self.current_folds_list) == 0:
            self.prun = False

        self.current_folds_list.append(value)

        self._populate_best_folds_at_first_run(value)

    def decide_prun(self):

        fold_num = len(self.current_folds_list)
        mean_best_folds = sum(self.best_folds_list[:fold_num]) / fold_num
        mean_curr_folds = sum(self.current_folds_list) / fold_num

        if len(self.current_folds_list) >= 2:
            if mean_best_folds * self.tolerance_scaler < mean_curr_folds:
                self.prun = True
                self.value_to_return = mean_curr_folds
                self.current_folds_list = []

        if fold_num == self.n_folds and mean_best_folds > mean_curr_folds:
            self.best_folds_list = self.current_folds_list
            self.current_folds_list = []
        elif len(self.current_folds_list) == self.n_folds:
            self.current_folds_list = []
