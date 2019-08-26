from prunedcv import PrunedCV, PrunedGridSearchCV
from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import pytest


def test_pruner_prun_yes():
    
    pruner = PrunedCV(4, 0.1)

    for i in range(6):
        pruner._add_split_value_and_prun(1.0)

    pruner._add_split_value_and_prun(10000.0)

    assert pruner.prune


def test_pruner_prun_no():

    pruner = PrunedCV(4, 0.1)

    for i in range(4):
        pruner._add_split_value_and_prun(1.0)

    for i in range(3):
        pruner._add_split_value_and_prun(.6)

    assert not pruner.prune


def test_pruner_prun_back():

    pruner = PrunedCV(4, 0.1)

    for i in range(4):
        pruner._add_split_value_and_prun(1.0)

    for i in range(2):
        pruner._add_split_value_and_prun(10000.0)

    for i in range(3):
        pruner._add_split_value_and_prun(1.0)

    assert not pruner.prune


def test_prun_first_run():

    pruner = PrunedCV(4, 0.1)

    for i in range(4):
        pruner._add_split_value_and_prun(1.0)

    assert pruner.best_splits_list_ == [1.0, 1.0, 1.0, 1.0]


def test_prun_first_run_check():

    pruner = PrunedCV(4, 0.1)

    for i in range(4):
        pruner._add_split_value_and_prun(1.0)

    assert not pruner.first_run_


def test_prun_folds_int():

    with pytest.raises(TypeError):
        pruner = PrunedCV(1.1, 0.1)
        pruner._add_split_value_and_prun(1)


def test_prun_folds_num():

    with pytest.raises(ValueError):
        pruner = PrunedCV(1, 0.1)
        pruner._add_split_value_and_prun(1)


def test_prun_vals_type():

    with pytest.raises(TypeError):
        pruner = PrunedCV(4, 0.1)
        pruner._add_split_value_and_prun(1)


def test_prun_score_val_constant():

    pruner = PrunedCV(4, 0.1)

    for i in range(8):
        pruner._add_split_value_and_prun(1.0)

    assert pruner.cross_val_score_value == 1.0


def test_prun_score_val_dec():

    pruner = PrunedCV(4, 0.1)

    for i in range(7):
        pruner._add_split_value_and_prun(1.0)

    pruner._add_split_value_and_prun(.9)

    assert pruner.cross_val_score_value < 1.0


def test_prun_score_val_inc():

    pruner = PrunedCV(4, 0.1)

    for i in range(7):
        pruner._add_split_value_and_prun(1.0)

    pruner._add_split_value_and_prun(1.1)

    assert pruner.cross_val_score_value > 1.0


def test_prun_score_val_best():

    pruner = PrunedCV(4, 0.1)

    for i in range(7):
        pruner._add_split_value_and_prun(1.0)

    pruner._add_split_value_and_prun(1.1)

    assert sum(pruner.best_splits_list_) / pruner.cv == 1.0


def test_prun_pruned_cv_score():

    pruner = PrunedCV(4, 0.1)

    for i in range(4):
        pruner._add_split_value_and_prun(1.0)

    for i in range(2):
        pruner._add_split_value_and_prun(2.0)

    assert pruner.cross_val_score_value == 2.0


def test_prun_3models():

    data = fetch_california_housing()
    x = data['data']
    y = data['target']

    pruner = PrunedCV(cv=8, tolerance=.1)

    model1 = LGBMRegressor(max_depth=25)
    model2 = LGBMRegressor(max_depth=10)
    model3 = LGBMRegressor(max_depth=2)

    score1 = pruner.cross_val_score(model1, x, y, shuffle=True, random_state=42)
    score2 = pruner.cross_val_score(model2, x, y, shuffle=True, random_state=42)
    score3 = pruner.cross_val_score(model3, x, y, shuffle=True, random_state=42)

    assert (sum(pruner.best_splits_list_) / pruner.cv == score2) and (score2 < score1) and (score2 < score3)


def test_prun_cv_x():

    with pytest.raises(TypeError):
        pruner = PrunedCV(cv=4, tolerance=.1)

        model = LGBMRegressor()
        x = [1, 2, 3]
        y = np.array([1, 2, 3])
        pruner.cross_val_score(model, x, y)


def test_prun_cv_y():

    with pytest.raises(TypeError):
        pruner = PrunedCV(cv=4, tolerance=.1)

        model = LGBMRegressor()
        y = [1, 2, 3]
        x = np.array([1, 2, 3])
        pruner.cross_val_score(model, x, y)


def test_prun_cv_xy():

    with pytest.raises(TypeError):
        pruner = PrunedCV(cv=4, tolerance=.1)

        model = LGBMRegressor()
        y = [1, 2, 3]
        x = [1, 2, 3]
        pruner.cross_val_score(model, x, y)


def test_prun_cv_x_df():

    data = fetch_california_housing()
    x = pd.DataFrame(data['data'])
    y = data['target']

    pruner = PrunedCV(cv=8, tolerance=.1)

    model = LGBMRegressor()

    pruner.cross_val_score(model, x, y)

    assert len(pruner.best_splits_list_) == pruner.cv


def test_prun_cv_xy_df_ser():

    data = fetch_california_housing()
    x = pd.DataFrame(data['data'])
    y = pd.Series(data['target'])

    pruner = PrunedCV(cv=8, tolerance=.1)

    model = LGBMRegressor()

    pruner.cross_val_score(model, x, y)

    assert len(pruner.best_splits_list_) == pruner.cv


def test_prun_cv_y_ser():

    data = fetch_california_housing()
    x = data['data']
    y = pd.Series(data['target'])

    pruner = PrunedCV(cv=8, tolerance=.1)

    model = LGBMRegressor()

    pruner.cross_val_score(model, x, y)

    assert len(pruner.best_splits_list_) == pruner.cv


def test_prun_set_tolerance_1():

    with pytest.raises(TypeError):
        pruner = PrunedCV(4, 0.1)
        pruner.set_tolerance(1)


def test_prun_set_tolerance_2():

    with pytest.raises(ValueError):
        pruner = PrunedCV(4, 0.1)
        pruner.set_tolerance(-1.0)


def test_prun_cv_metric():

    with pytest.raises(ValueError):

        data = fetch_california_housing()
        x = data['data']
        y = pd.Series(data['target'])

        pruner = PrunedCV(4, 0.1)

        model = LGBMRegressor()

        pruner.cross_val_score(model, x, y, metric='rmsle')


def test_pruner_mae():

    data = fetch_california_housing()
    x = data['data']
    y = pd.Series(data['target'])

    pruner = PrunedCV(4, 0.1)

    model = LGBMRegressor(objective='mae')

    pruner.cross_val_score(model, x, y, metric='mae')


def test_pruner_higher_value1():

    pruner = PrunedCV(4, 0.1)

    assert pruner._significantly_higher_value(1.0, 2.0, True, .1)


def test_pruner_higher_value2():

    pruner = PrunedCV(4, 0.1)

    assert not pruner._significantly_higher_value(1.0, 1.05, True, .1)


def test_pruner_higher_value3():

    pruner = PrunedCV(4, 0.1)

    assert not pruner._significantly_higher_value(-1.0, -1.05, False, .1)


def test_pruner_higher_value4():

    pruner = PrunedCV(4, 0.1)

    assert pruner._significantly_higher_value(-1.0, -0.8, False, .1)


def test_pruner_pgs():

    data = fetch_california_housing()
    x = data['data']
    y = data['target']

    model = LGBMRegressor()

    params_grid = {'max_depth': [25, 10, 2]}

    pgs = PrunedGridSearchCV(estimator=model,
                             params_grid=params_grid,
                             cv=8,
                             tolerance=0.1,
                             shuffle=True,
                             random_state=42)

    pgs.fit(x, y)

    assert pgs.best_params['max_depth'] == 10


def test_prun_first_run_list_len():
    data = fetch_california_housing()
    x = data['data']
    y = data['target']

    pruner = PrunedCV(8, 0.0)

    model = LGBMRegressor(n_estimators=2)

    pruner.cross_val_score(model, x, y)

    assert len(pruner.best_splits_list_) == 8


def test_prun_accuracy():
    x, y = load_wine(return_X_y=True)

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)
    params_grid = {'C': [0.001, 1.0, 0.01]}
    pgs = PrunedGridSearchCV(model, params_grid, cv=4, scoring='accuracy', minimize=False)

    pgs.fit(x, y)
    assert pgs.best_params['C'] == 1.0


def test_pgs_auc_weighted():
    x, y = load_wine(return_X_y=True)
    x = x[y != 2]
    y = y[y != 2]
    np.random.seed(42)
    w = np.random.rand(len(y)) * 2

    model = LogisticRegression(solver='lbfgs', max_iter=10000)
    params_grid = {'C': [0.001, 1.0, 0.01]}
    pgs = PrunedGridSearchCV(model, params_grid, cv=4, scoring='auc', minimize=False)

    pgs.fit(x, y, sample_weight=w)

    assert round(pgs.best_score, 4) == -0.9953
    assert pgs.best_params['C'] == 1.0
