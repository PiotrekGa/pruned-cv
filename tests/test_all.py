from prunedcv import PrunerCV
from sklearn.datasets import fetch_california_housing
from lightgbm import LGBMRegressor
import numpy as np
import pytest


def test_pruner_prun_yes():
    
    pruner = PrunerCV(4, 0.1)

    for i in range(6):
        pruner.add_split_value_and_prun(1.0)

    pruner.add_split_value_and_prun(10000.0)

    assert pruner.prun


def test_pruner_prun_no():

    pruner = PrunerCV(4, 0.1)

    for i in range(4):
        pruner.add_split_value_and_prun(1.0)

    for i in range(3):
        pruner.add_split_value_and_prun(.6)

    assert not pruner.prun


def test_pruner_prun_back():

    pruner = PrunerCV(4, 0.1)

    for i in range(4):
        pruner.add_split_value_and_prun(1.0)

    for i in range(2):
        pruner.add_split_value_and_prun(10000.0)

    for i in range(3):
        pruner.add_split_value_and_prun(1.0)

    assert not pruner.prun


def test_prun_first_run():

    pruner = PrunerCV(4, 0.1)

    for i in range(4):
        pruner.add_split_value_and_prun(1.0)

    assert pruner.best_splits_list == [1.0, 1.0, 1.0, 1.0]


def test_prun_first_run_check():

    pruner = PrunerCV(4, 0.1)

    for i in range(4):
        pruner.add_split_value_and_prun(1.0)

    assert not pruner.first_run


def test_prun_folds_int():

    with pytest.raises(TypeError):
        pruner = PrunerCV(1.1, 0.1)
        pruner.add_split_value_and_prun(1)


def test_prun_folds_num():

    with pytest.raises(ValueError):
        pruner = PrunerCV(1, 0.1)
        pruner.add_split_value_and_prun(1)


def test_prun_vals_type():

    with pytest.raises(TypeError):
        pruner = PrunerCV(4, 0.1)
        pruner.add_split_value_and_prun(1)


def test_prun_score_val_constant():

    pruner = PrunerCV(4, 0.1)

    for i in range(8):
        pruner.add_split_value_and_prun(1.0)

    assert pruner.cross_val_score == 1.0


def test_prun_score_val_dec():

    pruner = PrunerCV(4, 0.1)

    for i in range(7):
        pruner.add_split_value_and_prun(1.0)

    pruner.add_split_value_and_prun(.9)

    assert pruner.cross_val_score < 1.0


def test_prun_score_val_inc():

    pruner = PrunerCV(4, 0.1)

    for i in range(7):
        pruner.add_split_value_and_prun(1.0)

    pruner.add_split_value_and_prun(1.1)

    assert pruner.cross_val_score > 1.0


def test_prun_score_val_best():

    pruner = PrunerCV(4, 0.1)

    for i in range(7):
        pruner.add_split_value_and_prun(1.0)

    pruner.add_split_value_and_prun(1.1)

    assert sum(pruner.best_splits_list) / pruner.n_splits == 1.0


def test_prun_pruned_cv_score():

    pruner = PrunerCV(4, 0.1)

    for i in range(4):
        pruner.add_split_value_and_prun(1.0)

    for i in range(2):
        pruner.add_split_value_and_prun(2.0)

    assert pruner.cross_val_score == 2.0


def test_prun_3models():

    data = fetch_california_housing()
    x = data['data']
    y = data['target']

    pruner = PrunerCV(n_splits=8, tolerance=.1)

    model1 = LGBMRegressor(max_depth=25)
    model2 = LGBMRegressor(max_depth=10)
    model3 = LGBMRegressor(max_depth=2)

    pruner.cross_validate_score(model1, x, y, shuffle=True)
    pruner.cross_validate_score(model2, x, y, shuffle=True)
    pruner.cross_validate_score(model3, x, y, shuffle=True)

    assert pruner.best_model.get_params()['max_depth'] == 10


def test_prun_cv_x():

    with pytest.raises(TypeError):
        pruner = PrunerCV(n_splits=4, tolerance=.1)

        model = LGBMRegressor()
        x = [1, 2, 3]
        y = np.array([1, 2, 3])
        pruner.cross_validate_score(model, x, y)


def test_prun_cv_y():

    with pytest.raises(TypeError):
        pruner = PrunerCV(n_splits=4, tolerance=.1)

        model = LGBMRegressor()
        y = [1, 2, 3]
        x = np.array([1, 2, 3])
        pruner.cross_validate_score(model, x, y)


def test_prun_cv_xy():

    with pytest.raises(TypeError):
        pruner = PrunerCV(n_splits=4, tolerance=.1)

        model = LGBMRegressor()
        y = [1, 2, 3]
        x = [1, 2, 3]
        pruner.cross_validate_score(model, x, y)
