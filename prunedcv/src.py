from sklearn.model_selection import KFold, StratifiedKFold, ParameterGrid, ParameterSampler
from sklearn import metrics
import numpy
import pandas


class PrunedRandomizedSearchCV:

    """PrunedRandomizedSearchCV is a pruned version of scikit-learn RandomizedSearchCV.

    It applies pruning methodology by using PrunedCV.

    Args:
        estimator:
            An estimator to calculate cross-validated score
        param_distributions:
            A distribution space of hyperparameters for sampling.
            Please use distributions available in
            scipy.stats.distributions module
        n_iter:
            Number of hyperparameter samples to be evaluated
            with cross-validation
        cv:
            Number of folds to be created for cross-validation
        tolerance:
            Default = 0.1.
            The value creates boundary
            around the best score.
            If ongoing scores are outside the boundary,
            the trial is pruned.
        scoring:
            Default = 'mse'
            Metric from scikit-learn metrics to be optimized.
        splits_to_start_pruning:
            Default = 2
            The fold at which pruning may be first applied.
        minimize:
            Default = True
            The direction of the optimization.
        shuffle:
            Default = False
            If True, shuffle the data before splitting them into folds.
        random_state:
            Default = None
            If any integer value, creates a seed for random number generation.

        Usage example:

            from scipy.stats.distributions import uniform, randint
            from sklearn.datasets import fetch_california_housing
            from prunedcv import PrunedRandomizedSearchCV
            from lightgbm import LGBMRegressor

            data = fetch_california_housing()
            x = data['data']
            y = data['target']

            model = LGBMRegressor()
            params_grid = {'n_estimators': randint(2,100),
                            'max_depth': randint(2,200),
                            'learning_rate': uniform(.001, .2)}

            prs = PrunedRandomizedSearchCV(model,
                                           param_distributions=params_grid,
                                           n_iter=100,
                                           cv=12,
                                           tolerance=0.1,
                                           random_state=42)
            prs.fit(x, y)
            prs.best_params
            """

    def __init__(self,
                 estimator,
                 param_distributions,
                 n_iter,
                 cv,
                 tolerance=0.1,
                 scoring='mse',
                 splits_to_start_pruning=2,
                 minimize=True,
                 shuffle=False,
                 random_state=None):

        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.tolerance = tolerance
        self.scoring = scoring
        self.splits_to_start_pruning = splits_to_start_pruning
        self.minimize = minimize
        self.shuffle = shuffle
        self.random_state = random_state
        self.params_grid_iterable = ParameterSampler(param_distributions=self.param_distributions,
                                                     n_iter=self.n_iter,
                                                     random_state=self.random_state)
        self.best_params = None
        self.best_score = None

    def fit(self, x, y):

        """Executes search within the hyperparameter space.

        Args:
            x:
                numpy ndarray or pandas DataFrame
            y:
                numpy ndarray or pandas Series
        """

        pruner = PrunedCV(self.cv,
                          self.tolerance,
                          self.splits_to_start_pruning,
                          self.minimize)

        for params_set in self.params_grid_iterable:
            self.estimator.set_params(**params_set)
            score = pruner.cross_val_score(self.estimator,
                                           x,
                                           y,
                                           metric=self.scoring,
                                           shuffle=self.shuffle,
                                           random_state=self.random_state)

            if self.best_score is not None:
                if self.best_score > score:
                    self.best_score = score
                    self.best_params = params_set
            else:
                self.best_score = score
                self.best_params = params_set


class PrunedGridSearchCV:

    """PrunedGridSearchCV is a pruned version of scikit-learn GridSearchCV.

    It applies pruning methodology by using PrunedCV.

    Args:
        estimator:
            An estimator to calculate cross-validated score
        params_grid:
            Dict of hypermarameters to be scored.
        cv:
            Number of folds to be created for cross-validation
        tolerance:
            Default = 0.1.
            The value creates boundary
            around the best score.
            If ongoing scores are outside the boundary,
            the trial is pruned.
        scoring:
            Default = 'mse'
            Metric from scikit-learn metrics to be optimized.
        splits_to_start_pruning:
            Default = 2
            The fold at which pruning may be first applied.
        minimize:
            Default = True
            The direction of the optimization.
        shuffle:
            Default = False
            If True, shuffle the data before splitting them into folds.
        random_state:
            Default = None
            If any integer value, creates a seed for random number generation.

        Usage example:

            from lightgbm import LGBMRegressor
            from sklearn.datasets import fetch_california_housing
            from prunedcv import PrunedGridSearchCV

            data = fetch_california_housing()
            x = data['data']
            y = data['target']

            model = LGBMRegressor()
            params_grid = {'n_estimators': [2, 5, 10, 20, 50, 100],
                            'max_depth': [2,5,10,20,50,100,200],
                            'learning_rate': [.001, .002, .005, .01, .02, .05, .1, .2]}

            pgs = PrunedGridSearchCV(model,
                                     params_grid,
                                     cv=12,
                                     tolerance=0.1)
            pgs.fit(x, y)
            pgs.best_params
            """

    def __init__(self,
                 estimator,
                 params_grid,
                 cv,
                 tolerance=0.1,
                 scoring='mse',
                 splits_to_start_pruning=2,
                 minimize=True,
                 shuffle=False,
                 random_state=None):

        self.estimator = estimator
        self.params_grid = params_grid
        self.cv = cv
        self.tolerance = tolerance
        self.scoring = scoring
        self.splits_to_start_pruning = splits_to_start_pruning
        self.minimize = minimize
        self.shuffle = shuffle
        self.random_state = random_state
        self.params_grid_iterable = ParameterGrid(self.params_grid)
        self.best_params = None
        self.best_score = None

    def fit(self, x, y):

        """Executes search within the hyperparameter space.

        Args:
            x:
                numpy ndarray or pandas DataFrame
            y:
                numpy ndarray or pandas Series
        """

        pruner = PrunedCV(self.cv,
                          self.tolerance,
                          self.splits_to_start_pruning,
                          self.minimize)

        for params_set in self.params_grid_iterable:
            self.estimator.set_params(**params_set)
            score = pruner.cross_val_score(self.estimator,
                                           x,
                                           y,
                                           metric=self.scoring,
                                           shuffle=self.shuffle,
                                           random_state=self.random_state)

            if self.best_score is not None:
                if self.best_score > score:
                    self.best_score = score
                    self.best_params = params_set
            else:
                self.best_score = score
                self.best_params = params_set


class PrunedCV:

    """PrunedCV applied pruning to cross-validation. Based on scores
    from initial splits (folds) is decides whether it's worth to
    continue the cross-validation. If not it stops the process and returns
    estimated final score.

    If the trial is worth checking (the initial scores are
    better than the best till the time or withing tolerance border) it's equivalent
    to standard cross-validation. Otherwise the trial is pruned.


    Args:
        cv:
            Number of folds to be created for cross-validation
        tolerance:
            Default = 0.1.
            The value creates boundary
            around the best score.
            If ongoing scores are outside the boundary,
            the trial is pruned.
        splits_to_start_pruning:
            Default = 2
            The fold at which pruning may be first applied.
        minimize:
            Default = True
            The direction of the optimization.

    Usage example:

        from lightgbm import LGBMRegressor
        from sklearn.datasets import fetch_california_housing
        from prunedcv import PrunedCV
        import numpy as np

        data = fetch_california_housing()
        x = data['data']
        y = data['target']

        pruner = PrunedCV(cv=8, tolerance=.1)

        model1 = LGBMRegressor(max_depth=25)
        model2 = LGBMRegressor(max_depth=10)
        model3 = LGBMRegressor(max_depth=2)

        pruner.cross_val_score(model1, x, y)
        pruner.cross_val_score(model2, x, y)
        pruner.cross_val_score(model3, x, y)

        print('best score: ', round(sum(pruner.best_splits_list_) / len(pruner.best_splits_list_),4))
            """

    def __init__(self,
                 cv,
                 tolerance=0.1,
                 splits_to_start_pruning=2,
                 minimize=True):

        if not isinstance(cv, int):
            raise TypeError
        if cv < 2:
            raise ValueError

        self.cv = cv
        self.set_tolerance(tolerance)
        self.splits_to_start_pruning = splits_to_start_pruning
        self.minimize = minimize
        self.prune = False
        self.cross_val_score_value = None
        self.current_splits_list_ = []
        self.best_splits_list_ = []
        self.first_run_ = True

    def set_tolerance(self,
                      tolerance):
        """Set tolerance value

        Args:
            tolerance:
            The value creates boundary
            around the best score.
            If ongoing scores are outside the boundary,
            the trial is pruned.
        """

        if not isinstance(tolerance, float):
            raise TypeError
        if tolerance < 0:
            raise ValueError

        self.tolerance = tolerance

    def cross_val_score(self,
                        model,
                        x,
                        y,
                        metric='mse',
                        shuffle=False,
                        random_state=None):

        """Calculates pruned scores

        Args:
            model:
                An estimator to calculate cross-validated score
            x:
                numpy ndarray or pandas DataFrame
            y:
                numpy ndarray or pandas Series
            metric:
                Default = 'mse'
                Metric from scikit-learn metrics to be optimized.
            shuffle:
                Default = False
                If True, shuffle the data before splitting them into folds.
            random_state:
                Default = None
                If any integer value, creates a seed for random number generation.

        Usage example:

            Check PrunedCV use example.
        """

        if not isinstance(x, (numpy.ndarray, pandas.core.frame.DataFrame)):
            raise TypeError

        if not isinstance(y, (numpy.ndarray, pandas.core.series.Series)):
            raise TypeError

        if metric not in ['mse',
                          'mae',
                          'accuracy']:
            raise ValueError

        if metric in ['mse',
                      'mae']:
            kf = KFold(n_splits=self.cv,
                       shuffle=shuffle,
                       random_state=random_state)

        elif metric in ['accuracy']:

            kf = StratifiedKFold(n_splits=self.cv,
                                 shuffle=shuffle,
                                 random_state=random_state)

        else:
            raise ValueError

        for train_idx, test_idx in kf.split(x, y):
            if not self.prune:

                if isinstance(x, numpy.ndarray):
                    x_train = x[train_idx]
                    x_test = x[test_idx]
                else:
                    x_train = x.iloc[train_idx, :]
                    x_test = x.iloc[test_idx, :]
                if isinstance(y, numpy.ndarray):
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                else:
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]

                model.fit(x_train, y_train)
                y_test_teor = model.predict(x_test)

                if metric == 'mse':
                    self._add_split_value_and_prun(metrics.mean_squared_error(y_test,
                                                                              y_test_teor))
                elif metric == 'mae':
                    self._add_split_value_and_prun(metrics.mean_absolute_error(y_test,
                                                                               y_test_teor))
                elif metric == 'accuracy':
                    self._add_split_value_and_prun(metrics.accuracy_score(y_test,
                                                                          y_test_teor))

        self.prune = False
        return self.cross_val_score_value

    def _add_split_value_and_prun(self,
                                  value):

        if not isinstance(value, float):
            raise TypeError

        if len(self.current_splits_list_) == 0:
            self.prune = False

        if self.minimize:
            self.current_splits_list_.append(value)
        else:
            self.current_splits_list_.append(-value)

        if self.first_run_:
            self._populate_best_splits_list_at_first_run(value)
        else:
            self._decide_prune()

        if len(self.current_splits_list_) == self.cv:
            self._serve_last_split()

    def _populate_best_splits_list_at_first_run(self,
                                                value):

        if self.minimize:
            self.best_splits_list_.append(value)
        else:
            self.best_splits_list_.append(-value)

        if len(self.best_splits_list_) == self.cv:
            self.first_run_ = False

    def _decide_prune(self):

        split_num = len(self.current_splits_list_)
        mean_best_splits = sum(self.best_splits_list_[:split_num]) / split_num
        mean_curr_splits = sum(self.current_splits_list_) / split_num

        if self.cv > split_num >= self.splits_to_start_pruning:

            self.prune = self._significantly_higher_value(mean_best_splits,
                                                          mean_curr_splits,
                                                          self.minimize,
                                                          self.tolerance)

            if self.prune:
                self.cross_val_score_value = self._predict_pruned_score(mean_curr_splits,
                                                                        mean_best_splits)
                self.current_splits_list_ = []

    @staticmethod
    def _significantly_higher_value(mean_best_splits,
                                    mean_curr_splits,
                                    minimize,
                                    tolerance):
        tolerance_scaler_if_min = 1 + minimize * tolerance
        tolerance_scaler_if_max = 1 + (1 - minimize) * tolerance
        return mean_best_splits * tolerance_scaler_if_min < mean_curr_splits * tolerance_scaler_if_max

    def _predict_pruned_score(self,
                              mean_curr_splits,
                              mean_best_splits):
        return (mean_curr_splits / mean_best_splits) * (sum(self.best_splits_list_) / self.cv)

    def _serve_last_split(self):

        if sum(self.best_splits_list_) > sum(self.current_splits_list_):
            self.best_splits_list_ = self.current_splits_list_

        self.cross_val_score_value = sum(self.current_splits_list_) / self.cv
        self.current_splits_list_ = []
