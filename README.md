# pruned-cv

## Introduction
The package implements pruned cross-validation technique, which verifies whether next folds are worth calculating.
It's components may be used as a standalone methods or as a part of hyperparameter optimization frameworks like 
Hyperopt or Optuna.

The package proved to be over two times faster than Scikit-Learn GridSearchCV yielding the same results.

## Installation
Just clone the repository:

`git clone git@github.com:PiotrekGa/pruned-cv.git`

and install it with pip:

`pip install -e .`

## Examples

You can find example notebooks in examples section of the repository.

#### GridSearchCV benchmark

https://github.com.PiotrekGa/pruned-cv/examples/GridSearchCV_Benchmark.ipynb