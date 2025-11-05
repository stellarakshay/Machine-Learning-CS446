"""
CS 446/ECE 449 - Model Selection Homework
==========================================
In this assignment, you will implement model selection using k-fold cross-validation
to find the best hyperparameters for polynomial regression with regularization.

Instructions:
- Complete all TODO sections
- Do not modify the function signatures
- You may add helper functions if needed
"""

from hw4_utils import (
    ModelPipeline,
    create_polynomial_features
)

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')


def cross_validate_model(X, y, model, k_folds=5):
    """
    Perform k-fold cross-validation and return average validation error.
    
    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        model: Sklearn model object
        k_folds: Number of folds for cross-validation
    
    Returns:
        avg_val_error: Average validation MSE across all folds
        std_val_error: Standard deviation of validation MSE across folds
    """
    # TODO: Implement k-fold cross-validation
    # 1. Create KFold() object with k_folds splits (use shuffle=True, random_state=42)
    # 2. For each fold:
    #    - Split data into train and validation sets
    #    - Fit model on training data
    #    - Calculate MSE on validation data
    # 3. Return average and standard deviation of validation errors
    
    # Remark 1: for `model`, you can safely assume that you can call model.fit(X, y) to 
    #   train the model on data X, y; in addition, you can call model.predict(X)
    #   to obtain predictions from the model.
    # Remark 2: for each iteration during k fold validation, please do 
    #   `model_clone = deepcopy(model)` and call `model_clone.fit()` and `model_clone.predict()`. 
    #   Otherwise, you will be training a model that is from the previous iteration.
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_mses = []

    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model_copy = deepcopy(model)
        model_copy.fit(X_tr, y_tr)

        y_hat = model_copy.predict(X_va)
        # Compute MSE manually to avoid extra imports
        fold_mses.append(float(np.mean((y_va - y_hat) ** 2)))

    mean_val_error = float(np.mean(fold_mses))
    std_val_error  = float(np.std(fold_mses))

    return mean_val_error, std_val_error


def select_best_model(X_train, y_train):
    """
    Select the best model and hyperparameters using cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        returned_best_model: Trained best model
    """
    # TODO Implement model selection
    # 1. For each polynomial degree:
    #    a. Create polynomial features for training data (already implemented)
    #    b. Standardize features using StandardScaler (fit on train, transform both) (already implemented)
    #    c. For LinearRegression: 
    #       - Perform cross-validation with k = 5
    #    d. For Ridge regression: 
    #       - Try different alpha values
    #       - Perform cross-validation for each alpha with k = 5
    #    e. For Lasso regression: 
    #       - Try different alpha values
    #       - Perform cross-validation for each alpha with k = 5
    # 2. Select the best model based on lowest cross-validation error

    # Remark 1: you can use `LinearRegression()` to initialize the Linear Regression model.
    # Remark 2: you can use `Ridge(alpha=alpha, random_state=42)` to initialize the Ridge 
    #   Regression model.
    # Remark 3: you can use `Lasso(alpha=alpha, random_state=42, max_iter=2000)` to 
    #   initialize the Lasso Regression model.
    
    # Hyperparameter search space (Do not modify these!)
    degrees = [1, 2, 3, 4, 5, 6, 7, 8]
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    best_mean = float("inf")
    best_degree = None
    best_model = None

    for degree in degrees:
        # Linear Regression (no alpha grid)
        lr_pipeline = ModelPipeline(degree, LinearRegression(), StandardScaler())
        lr_mean, lr_std = cross_validate_model(X_train, y_train, lr_pipeline, k_folds=5)
        if lr_mean < best_mean:
            best_mean = lr_mean
            best_degree = degree
            best_model = LinearRegression()

        # Ridge over alphas
        for alpha in alphas:
            ridge_pipeline = ModelPipeline(degree, Ridge(alpha=alpha, random_state=42), StandardScaler())
            rg_mean, rg_std = cross_validate_model(X_train, y_train, ridge_pipeline, k_folds=5)
            if rg_mean < best_mean:
                best_mean = rg_mean
                best_degree = degree
                best_model = Ridge(alpha=alpha, random_state=42)

        # Lasso over alphas
        for alpha in alphas:
            lasso_pipeline = ModelPipeline(degree, Lasso(alpha=alpha, random_state=42, max_iter=2000), StandardScaler())
            ls_mean, ls_std = cross_validate_model(X_train, y_train, lasso_pipeline, k_folds=5)
            if ls_mean < best_mean:
                best_mean = ls_mean
                best_degree = degree
                best_model = Lasso(alpha=alpha, random_state=42, max_iter=2000)

    # Fit and return the winning pipeline on all training data
    returned_best_model = ModelPipeline(best_degree, best_model, StandardScaler())
    returned_best_model.fit(X_train, y_train)
    return returned_best_model