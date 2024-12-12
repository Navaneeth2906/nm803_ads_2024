# This file contains code for suporting addressing questions in the data


import matplotlib.pyplot as plt
import pandas as pd 

import statsmodels.api as sm
import numpy as np

from shapely.geometry import Point


"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

def create_model(oa_tag_counts_dists_gdf, features, response):
    X = oa_tag_counts_dists_gdf[features]

    X = sm.add_constant(X)

    y = oa_tag_counts_dists_gdf[response]

    model = sm.OLS(y, X)
    results = model.fit()

    y_pred = results.predict(X)

    print(f"Correlation: {np.corrcoef(y, y_pred)[0, 1]}")

    plt.scatter(y, y_pred, label='Data Points', color='blue')

    plt.xlabel(f"Actual {response}")
    plt.ylabel(f"Predicted {response}")
    plt.show()

    print(results.summary())


    return y_pred, y, X


def create_model_regularised(oa_tag_counts_dists_gdf, features, response, alpha, L1_wt):
    X = oa_tag_counts_dists_gdf[features]

    X = sm.add_constant(X)

    y = oa_tag_counts_dists_gdf[response]

    model = sm.OLS(y, X)
    results = model.fit_regularized(alpha=alpha,L1_wt=L1_wt)

    y_pred = results.predict(X)

    print(f"Correlation: {np.corrcoef(y, y_pred)[0, 1]}")

    plt.scatter(y, y_pred, label='Data Points', color='blue')

    plt.xlabel(f"Actual {response}")
    plt.ylabel(f"Predicted {response}")
    plt.show()
    

    return y_pred, y, X