# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:24:56 2016
Script for running various classifiers
@author: afsar
This will run your code, you can make modifications to it as required 
for different components of the assignment as follows.
"""
from softKSVM import softKSVM
from kernels import kpoly,rbf
import matplotlib.pyplot as plt
import numpy as np
import itertools
if __name__ == '__main__':
    Xtr = np.array([[0,0],[0,1],[1,0],[1,1]]) #AND GATE
    Ytr = np.array([-1,1,1,-1])
    #P = softKSVM(kernel = kpoly, degree = 2, homo = 1, C=1000)
    P = softKSVM(kernel = rbf, degree = 2, homo = 1, C=1000)
    
    P.train(Xtr,Ytr)    
    Y = P.classify(Xtr)
    
    print ("Predicted Labels:",Y)
    print ("True Labels:",Ytr)
    print ("Error:", np.sum(Y!=Ytr)   ) 
    print ("Classifier:",P)
    
    ### PLOTTING CODE
    npts = 1000
    x = np.linspace(0,1,npts)
    y = np.linspace(0,1,npts)
    t = np.array(list(itertools.product(x,y)))
    z = P.score(t)
    z = np.reshape(z,(npts,npts))
    plt.imshow(z)    
    plt.contour(z,[-1,0,+1],linewidths = [1,2,1],colors=('k'),extent=[0,1.0,0,1.0], label='f(x)=0')
    plt.imshow(np.flipud(z), extent = [0,1.0,0,1.0], cmap=plt.cm.Purples); plt.colorbar()
    plt.scatter(Xtr[Ytr==1,0],Xtr[Ytr==1,1],marker = 's', c = 'r', s = 400)
    plt.scatter(Xtr[Ytr==-1,0],Xtr[Ytr==-1,1],marker = 'o',c = 'g', s = 400)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('$'+P.__str__()+'$')
    plt.axis([0.0,1.0,0.0,1.0])    
    plt.grid()
    plt.show()
    