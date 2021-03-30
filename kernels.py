# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:27:21 2016
Implementation of different kernels
@author: Bismillah Jan
"""
"""
This file contains implementations of different kernel functions.
I have implemented the polynomial kernel for your ease. 
You are to implement the RBF kernel. Please follow the signature of the function exactly.
"""
import numpy as np
from numpy import linalg as lg
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
def kpoly(X1,X2 = None,**kwargs):
    """
    Implementation of the polynomial kernel k(a,b) = (aTb + h)^p
    Input Arguments:
        X1 : n x d dimensional numpy array of n examples in d dimensions
        X2 : m x d dimensional numpy array of m examples in d dimensions
            optional, default X2 = X1
        optional keyword arguments:
            degree: the degree (p) of the polynomial, default is 1
            homo: whether the kernel is homogenous or not (h), default is 0
    Output Argumets:
        n x m dimensional numpy array (kernel or gram matrix)
    """
    p = kwargs.pop('degree',1)
    h = kwargs.pop('homo',0)    
    if X2 is None:
        X2 = X1
    K = (np.dot(X1,X2.T)+h)**p
    return K
	
def rbf(X1,X2 = None,**kwargs):
    """
    Implementation of the RBF kernel k(a,b) = exp(-gamma*||a-b||^2)
    Input Arguments:
        X1 : n x d dimensional numpy array of n examples in d dimensions
        X2 : m x d dimensional numpy array of m examples in d dimensions
            optional, default X2 = X1
        optional keyword arguments:
            gamma: speread control parameter of the RBF, default 1.0            
    Output Argumets:
        n x m dimensional numpy array (kernel or gram matrix)
    TODO: Implement the function
    """
    # To be implemented
    g=kwargs.pop('gamma',1.0)
    if X2 is None:
        X2 = X1    
    K=np.exp(-g*cdist(X1, X2, 'sqeuclidean'))
    return K
 
