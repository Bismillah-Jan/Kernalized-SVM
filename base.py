"""
Created on Wed Feb 03 22:01:45 2016
CIS 621 Machine Learning
Assignment 2
@author: <your name here>
@author-email: <your piazza email here>
stub developed by Dr. Fayyaz Minhas, DCIS, PIEAS, PAKISTAN
email: afsar at pieas dot edu dot pk
"""
import numpy as np
class linearClassifier:
    """
    Base class for linear classifier
    """
    def __init__(self,**kwargs):
        self.W = np.array([0])
        self.bias = 0
        
    def train(self,**kwargs):
        pass
    
    def score(self,x):
        """
        Return the discriminant score
        Input: x is m x d numpy array of m examples
        returns: m dimensional numpy vector of scores
        """
        return np.dot(x,self.W) + self.bias
        
    def classify(self,x):
        """
        Return the label score
        Input: x is m x d numpy array of m examples
        returns: m dimensional numpy vector of labels
        """
        return 2*(self.score(x)>=0) - 1
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
        
    def __str__(self):
        s = []
        for i,w in enumerate(self.W):
            si = "(%0.2f) x_%i" % (w,i+1)
            s.append(si)
        si = "(%0.2f)" % (self.bias)
        s.append(si)
        s = ' + '.join(s) 
        s += " = 0"
        return s