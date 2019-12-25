# multioutput-gbm  [![python versions](https://img.shields.io/badge/python-3.6+-blue.svg)](https://github.com/tanlab/multioutput-gbm)



Experimental Histogram Based Multi-Output Gradient Boosting Machines in Python.

An implementation of the paper [Exploiting random projections and sparsity with random forests and gradient boosting methods](https://arxiv.org/abs/1704.08067) on histogram based gradient boosting trees. 
Based on [pygbm](https://github.com/ogrisel/pygbm/)

## Installation

Use pip to install in "editable" mode:

    git clone https://github.com//tanlab/multioutput-gbm.git
    cd pygbm
    pip install -r requirements.txt
    pip install --editable .

Run the tests with pytest:

    pip install -r requirements.txt
    pytest

## Usage

To train multi-output data just use the fit() method, to predict from said model use predict_multi().

    from sklearn.datasets import make_regression
    from sklearn.metrics import r2_score
    from pygbm import GradientBoostingRegressor
    X, y = make_regression(random_state=0,n_targets=16)
    test = GradientBoostingRegressor().fit(X, y)
    predictions = test.predict_multi(X)
    print(r2_score(y, predictions, multioutput='uniform_average'))