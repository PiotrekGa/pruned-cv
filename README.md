# pruned-cv

## Introduction
The package implements pruned cross-validation technique, which verifies whether next folds are worth calculating.
It's components may be used as a standalone methods or as a part of hyperparameter optimization frameworks like 
Hyperopt or Optuna.

It proved to be over two times faster than Scikit-Learn GridSearchCV yielding the same results.

## Motivation

The idea was to improve speed of hyperparameter optimization. 
All the methods which base on cross-validation require many 
big folds number (8 is an absolute minimum) to assure that the surrogate model
(whether it's GridSearch, RandomSearch or a Bayesian model) does not overfit to the training set. 
Some hyperparameters set may be assessed of poor quality without calculating all the folds.

On the other hand Optuna proposes a mechanism of pruned learning for Artificial Neural Networks and 
Gradient Boosting Algorithms. It speeds the search process greatly but the issue with the method is that is prunes 
the trials based on a single validation sample. With relatively small datasets the model's quality 
variance may be high and lead to suboptimal hyperparameters choices.

Pruned-cv is a compromise between brut-force methods like GridSearch and more elaborate, but reluctant ones 
like Optuna's pruning.

## Installation

The package works with Python 3. To install it clone the repository:

`git clone git@github.com:PiotrekGa/pruned-cv.git`

and run:

`pip install -e pruned-cv`

## Examples

You can find example notebooks in examples section of the repository.

#### GridSearchCV benchmark

https://github.com.PiotrekGa/pruned-cv/examples/GridSearchCV_Benchmark.ipynb