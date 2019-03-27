# pruned-cv

## Introduction
The package implements pruned cross-validation technique, which verifies whether next folds are worth calculating. It's components may be used as a standalone methods or as a part of hyperparameter optimization frameworks like Hyperopt or Optuna.

It proved to be around three times faster than Scikit-Learn GridSearchCV and RandomizedSearchCV yielding the same results (see benchmarks section).

![gs_vs_pgs](https://raw.githubusercontent.com/PiotrekGa/PiotrekGa.github.io/master/images/gs_vs_pgs.png)

You can find a broader overview of the motivation an methodology under this
[directory](https://piotrekga.github.io/Pruned-Cross-Validation/) or alternatively on [Medium](https://towardsdatascience.com/pruned-cross-validation-for-hyperparameter-optimization-1c4e0588191a).

## Motivation

The idea was to improve speed of hyperparameter optimization. All the methods which base on cross-validation require many big folds number (8 is an absolute minimum) to assure that the surrogate model (whether it's GridSearch, RandomSearch or a Bayesian model) does not overfit to the training set. Some hyperparameters set may be assessed of poor quality without calculating all the folds.

On the other hand Optuna proposes a mechanism of pruned learning for Artificial Neural Networks and Gradient Boosting Algorithms. It speeds the search process greatly but the issue with the method is that is prunes the trials based on a single validation sample. With relatively small datasets the model's quality variance may be high and lead to suboptimal hyperparameters choices.

Pruned-cv is a compromise between brut-force methods like GridSearch and more elaborate, but vulnerable ones like Optuna's pruning.

## How does it work?

You can see example of correlations between cumulative scores on folds with the final score:

![correlations](https://raw.githubusercontent.com/PiotrekGa/PiotrekGa.github.io/master/images/correlations.png)

You may find the whole study notebook [here](https://github.com/PiotrekGa/pruned-cv/blob/master/examples/Correlations_between_folds.ipynb).

The package uses the fact that cumulative scores are highly correlated with the final score. 
In most cases after calculating 2 folds it's possible to predict the final score very accurately.
If the partial score is very poor the cross-validation is stopped (pruned) and the final scores value is predicted based on best till now scores.

## Installation

The package works with Python 3. To install it clone the repository:

`git clone git@github.com:PiotrekGa/pruned-cv.git`

and run:

`pip install -e pruned-cv`

## Examples

You can find example notebooks in the _examples_ section of the repository.

#### Usage with Optuna

https://github.com/PiotrekGa/pruned-cv/blob/master/examples/Usage_with_Optuna.ipynb

#### Usage with Hyperopt

https://github.com/PiotrekGa/pruned-cv/blob/master/examples/Usage_with_Hyperopt.ipynb

## Benchmarks

You can find benchmarks in examples section.

#### Grid Search CV

https://github.com/PiotrekGa/pruned-cv/blob/master/examples/GridSearchCV_Benchmark.ipynb

#### Randmized Search CV

https://github.com/PiotrekGa/pruned-cv/blob/master/examples/RandomizedSearchCV_Benchmark.ipynb
