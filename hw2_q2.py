import torch
import hw2_utils as utils
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split



def gaussian_theta(X, y):
    '''
    Arguments:
        X (S x N FloatTensor): features of each object
        y (S LongTensor): label of each object, y[i] = 0/1

    Returns:
        mu (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)

    '''
    pass

def gaussian_p(y):
    '''
    Arguments:
        y (S LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''
    pass

def gaussian_classify(mu,sigma2, p, X):
    '''
    Arguments:
        mu (2 x N Float Tensor): returned value #1 of `gaussian_MAP`
        sigma2 (2 x N Float Tensor): returned value #2 of `gaussian_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (S x N LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor): label of each object for classification, y[i] = 0/1
    
    '''
    pass
