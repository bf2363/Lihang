# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:31:59 2019

@author: bowei


    pd.options.display.float_format = '{:,.2f}'.format
    pd.set_option('display.width', 3000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 199)
"""
import logging
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

class perceptron(object):
    def __init__(self, eta=1, max_iter=1000, random_state=None, verbose=False):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self, X, Y, coef_init=None, intercept_init=None):
        if coef_init is None:
            coef_init = np.zeros(X.shape[1])
        if intercept_init is None:
            intercept_init = 0.
        
        np.random.seed(self.random_state)
        idxLst = list(np.random.randint(0,X.shape[0],size=self.max_iter))
        
        n_iter = 0
        
        while n_iter < self.max_iter:
            currIdx = idxLst.pop()
            currSign = Y[currIdx]*(X[currIdx].dot(coef_init) + intercept_init)
            if currSign <= 0:
                coef_init += self.eta * X[currIdx] * Y[currIdx]
                intercept_init += self.eta * Y[currIdx]
            n_iter += 1
            if self.verbose:
                logger.info(n_iter)
        
        self.coef = coef_init
        self.intercept = intercept_init
    
    def predict(self, x):
        return np.sign(x.dot(self.coef) + self.intercept)
    
        
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

X = X[Y < 2]
Y = Y[Y < 2]
Y = 2*Y -  1
xxx = perceptron(random_state=3,max_iter=200000,verbose=False, eta=0.01)
xxx.fit(X,Y)


plt.figure(figsize=(5, 4))
plt.scatter(X[:,0], X[:,1], c=Y)
x_vals = np.sort(np.unique(X[:,0]))
y_vals = -xxx.intercept/ xxx.coef[1] - xxx.coef[0] / xxx.coef[1]  * x_vals
plt.plot(x_vals, y_vals, '--') 
plt.show()


#========== 
from sklearn.linear_model import Perceptron
clf = Perceptron(tol=0., random_state=1,max_iter=50000,eta0=0.01)
clf.fit(X, Y)
plt.figure(figsize=(5, 4))
plt.scatter(X[:,0], X[:,1], c=Y)
x_vals = np.sort(np.unique(X[:,0]))
y_vals = -clf.intercept_/ clf.coef_[0][1] - clf.coef_[0][0] / clf.coef_[0][1]  * x_vals
plt.plot(x_vals, y_vals, '--') 
plt.show()
