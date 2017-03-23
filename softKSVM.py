# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:22:32 2016
Implementation of the soft kernelized SVM
@author: afsar
"""
"""
This file contains a stub implementation of the soft kernelized SVM. 
Your task is to implement the convex optimization through cvxopt in 
the train function and implement the score function of the SVM to utilize the kernels
"""
from base import linearClassifier
import numpy as np
from cvxopt import matrix, solvers
#from cvxopt.solvers import qp # Quadratic Programming Solver from CVXOPT
#from cvxopt import matrix 
from kernels import kpoly
import math
class softKSVM(linearClassifier):
    def __init__(self,**kwargs):
        """
        Initialization function:
        Optional Keyword arguments:
            kernel: handle to the kernel function, default is kpoly
                can be either a kernel function visible in scope or
                'matrix' in which case the code assumes that the kernel matrix
                is specified during training
                any keyword arguments to be passed to the kernel can also be 
                specified in this function.
            C: The C parameter of the SVM, default 1.0
                can either be a positive real number or a numpy vector if a
                separate value of this parameter is to be specified for each
                example               
        """
        self.kfun = kwargs.pop('kernel', kpoly)        
        self.C = kwargs.pop('C', 1.0) 
        self.cparam = kwargs
        linearClassifier.__init__(self,**kwargs)
        
    def train(self,Xtr,Ytr,**kwargs):
        """
        Training function 
            Xtr: m x d numpy array of m examples, the function expects it to be
                a kernel matrix if the kernel keyword argument in the initialization
                has been specified as 'matrix'.
            Ytr: m vector of training labels
        Key-word arguments
            None expected
        TO DO: Implement the training Function
        """
        if self.kfun == 'matrix': #if the kernel matrix is specified
            self.K = Xtr
        else: #otherwise call the kernel function 
           self. K = self.kfun(Xtr,**self.cparam) 
        assert self.K.shape[0] == self.K.shape[1]        #kernel matrix must be square
        N = self.K.shape[0]
        if not hasattr(self.C, "__len__"): #if C is not a list of numpy vector
            self.C = self.C*np.ones(N)
        assert len(self.C) == N #   now each example has a C value
        # Solve the QP Problem and obtain valid values for the following   
        self.alphas = np.zeros(N) #initialize alphas
        self.bias = 0.0
        K=self.K
        #________________________________________________________________
        self.rows=Xtr.shape[0]
        Ytr=Ytr.reshape((self.rows,1))
        P= (Ytr*np.transpose(Ytr))*K
        self.pp=P
        q=-1*np.ones(self.rows)
        
        ##___________________________________________
        #for -alpha<=0
        R_std = -1*np.eye(self.rows)
        s_std = np.zeros(self.rows)
        #--------------------------------------------        
        # alpha<= c
        R_slack = np.eye(self.rows)
        s_slack = self.C*np.ones(self.rows)
        #_____________________________________________
        #combining both constraints
        R = np.vstack((R_std, R_slack))
        s = np.vstack((s_std, s_slack))
        s=s.reshape((self.rows*2,1))
        ##___________________________________________
        
        U=Ytr.T
        v=0.0
        
        sol=solvers.qp(matrix(P, tc='d'), matrix(q), matrix(R), matrix(s), matrix(U, tc='d'),matrix(v)) 
        #qp(Q=P, p=q, G=R, h=s, A=U, b=v)        
        self.alphas=np.array(sol['x'])
        print ("Alphas are: ", self.alphas)
        
        self.svi=self.alphas>10**-6
        self.sv=np.zeros((self.svi.shape[0],Xtr.shape[1]))
        self.svLabels=np.zeros((self.svi.shape[0],1))
        for i in range(self.svi.shape[0]):
            self.sv[i]=Xtr[i]
            self.svLabels[i]=Ytr[i]
        #________________________________________________________________
        
        
    def score(self,x,**kwargs):
        """
        Implement the scoring function f(x) = sum_i(alpha_i * y_i * K(x,i)) + bias        
            x: numpy array of size m x d of m examples in d dimensions
        Return:
            A numpy array of scores of each example given as input
        TODO: Implement the scoring function for the SVM
        """
        if self.kfun != 'matrix' and len(self.sv):        
            k = self.kfun(x,self.sv,**self.cparam)
            #print "Kernel after test: ", k
        else:
            k = x
        
            
        self.W=self.alphas   
        self.mat=self.kfun(np.array([self.sv[1]]), self.sv,**self.cparam)        
        self.bias=self.svLabels[1]- np.dot((self.alphas*self.svLabels).T,self.mat.T)        
        z=np.dot((self.alphas*self.svLabels).T,k.T)+self.bias
            
        #print "bias: ", self.bias, "\nZ: ",z
        
       
        return z    
    
    