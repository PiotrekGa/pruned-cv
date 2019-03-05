from prunedcv.prunedcv import PrunerCV, ValueTooSmallError
import pytest


def test_pruner_prun_yes():
    pruner = PrunerCV(4, 0.1)
    for i in range(6):
        pruner.add_fold_value_and_prun(1.0)
    pruner.add_fold_value_and_prun(10000.0)
    assert pruner.prun


def test_pruner_prun_no():
    pruner = PrunerCV(4, 0.1)
    for i in range(4):
        pruner.add_fold_value_and_prun(1.0)
    for i in range(3):
        pruner.add_fold_value_and_prun(.6)
    assert not pruner.prun


def test_pruner_prun_back():
    pruner = PrunerCV(4, 0.1)
    for i in range(4):
        pruner.add_fold_value_and_prun(1.0)
    for i in range(2):
        pruner.add_fold_value_and_prun(10000.0)
    for i in range(3):
        pruner.add_fold_value_and_prun(1.0)
    assert not pruner.prun


def test_prun_first_run():
    pruner = PrunerCV(4, 0.1)
    for i in range(4):
        pruner.add_fold_value_and_prun(1.0)
    assert pruner.best_folds_list == [1.0, 1.0, 1.0, 1.0]


def test_prun_first_run_check():
    pruner = PrunerCV(4, 0.1)
    for i in range(4):
        pruner.add_fold_value_and_prun(1.0)
    assert not pruner.first_run


def test_prun_folds_int():
    with pytest.raises(TypeError):
        pruner = PrunerCV(1.1, 0.1)


def test_prun_folds_num():
    with pytest.raises(ValueTooSmallError):
        pruner = PrunerCV(1, 0.1)


def test_prun_vals_type():
    with pytest.raises(TypeError):
        pruner = PrunerCV(4, 0.1)
        pruner.add_fold_value_and_prun(1)


def test_prun_score_val_constant():
    pruner = PrunerCV(4, 0.1)
    for i in range(8):
        pruner.add_fold_value_and_prun(1.0)
    assert pruner.cross_val_score == 1.0


def test_prun_score_val_dec():
    pruner = PrunerCV(4, 0.1)
    for i in range(7):
        pruner.add_fold_value_and_prun(1.0)

    pruner.add_fold_value_and_prun(.9)
    assert pruner.cross_val_score < 1.0


def test_prun_score_val_inc():
    pruner = PrunerCV(4, 0.1)
    for i in range(7):
        pruner.add_fold_value_and_prun(1.0)
    pruner.add_fold_value_and_prun(1.1)
    assert pruner.cross_val_score > 1.0


def test_prun_score_val_best():
    pruner = PrunerCV(4, 0.1)
    for i in range(7):
        pruner.add_fold_value_and_prun(1.0)
    pruner.add_fold_value_and_prun(1.1)
    assert sum(pruner.best_folds_list) / pruner.n_folds == 1.0


def test_prun_pruned_cv_score():
    pruner = PrunerCV(4, 0.1)
    for i in range(4):
        pruner.add_fold_value_and_prun(1.0)
    for i in range(2):
        pruner.add_fold_value_and_prun(2.0)
    assert pruner.cross_val_score == 2.0
