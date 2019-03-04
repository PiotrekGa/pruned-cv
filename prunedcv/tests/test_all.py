from prunedcv.prunedcv.src import PrunerCV, ValueTooSmallError
import pytest


def test_pruner_prun_yes():
    pruner = PrunerCV(4, 0.1)

    for i in range(6):
        pruner.add_folds_value(1.0)
        pruner.decide_prun()

    pruner.add_folds_value(10000.0)
    pruner.decide_prun()

    assert pruner.prun


def test_pruner_prun_no():
    pruner = PrunerCV(4, 0.1)

    for i in range(4):
        pruner.add_folds_value(1.0)
        pruner.decide_prun()

    for i in range(3):
        pruner.add_folds_value(.6)
        pruner.decide_prun()

    assert not pruner.prun


def test_pruner_prun_back():
    pruner = PrunerCV(4, 0.1)

    for i in range(4):
        pruner.add_folds_value(1.0)
        pruner.decide_prun()

    for i in range(2):
        pruner.add_folds_value(10000.0)
        pruner.decide_prun()

    for i in range(3):
        pruner.add_folds_value(1.0)
        pruner.decide_prun()

    assert not pruner.prun


def test_prun_first_run():
    pruner = PrunerCV(4, 0.1)

    for i in range(4):
        pruner.add_folds_value(1.0)
        pruner.decide_prun()

    assert pruner.best_folds_list == [1.0, 1.0, 1.0, 1.0]


def test_prun_first_run_check():
    pruner = PrunerCV(4, 0.1)

    for i in range(4):
        pruner.add_folds_value(1.0)
        pruner.decide_prun()

    assert pruner.first_run == False


def test_prun_folds_int():
    with pytest.raises(TypeError):
        pruner = PrunerCV(1.1, 0.1)


def test_prun_folds_num():
    with pytest.raises(ValueTooSmallError):
        pruner = PrunerCV(1, 0.1)


def test_prun_vals_type():
    with pytest.raises(TypeError):
        pruner = PrunerCV(4, 0.1)
        pruner.add_folds_value(1)



