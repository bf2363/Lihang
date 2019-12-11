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
    def __init__(self, eta=1, max_iter=1000, random_state=None, verbose=False,tol=1e-5):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.tol = tol
    
    def lossFunc(self, X,Y, coef, intercept):
        L = (X.dot(coef)+intercept)*Y
        return -(L[L <= 0].sum())
    
    def fit(self, X, Y, coef_init=None, intercept_init=None):
        if coef_init is None:
            coef_init = np.array(X[0])
        if intercept_init is None:
            intercept_init = 0.
        
        np.random.seed(self.random_state)
        idxLst = list(np.random.randint(0,X.shape[0],size=self.max_iter))
        
        n_iter = 0
        
        L = self.lossFunc(X,Y, coef_init, intercept_init)

        while n_iter < self.max_iter:
            print "{0} iteration with EL {1}".format(n_iter,L)
            
            currIdx = idxLst.pop()
            currSign = Y[currIdx]*(X[currIdx].dot(coef_init) + intercept_init)
            if currSign <= 0:
                coef_init += self.eta * X[currIdx] * Y[currIdx]
                intercept_init += self.eta * Y[currIdx]
                L = self.lossFunc(X,Y, coef_init, intercept_init)
                if L <= 0:
                    break
            n_iter += 1
            if self.verbose:
                logger.info(n_iter)
        
        self.coef = coef_init
        self.intercept = intercept_init
    
    def predict(self, x):
        return np.sign(x.dot(self.coef) + self.intercept)
    
    

class perceptron_gram(object):
    def __init__(self, eta=1, max_iter=1000, random_state=None, verbose=False):
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
    
    def fit(self, X, Y):
        
        coef_init = np.zeros(X.shape[0])
        
        
        
        np.random.seed(self.random_state)
        idxLst = list(np.random.randint(0,X.shape[0],size=self.max_iter))
        
        self.Gram_matrix = X.dot(X.T)
        
        n_iter = 0
        
        while n_iter < self.max_iter:
            currIdx = idxLst.pop()
            currSign = Y[currIdx]*self.eta*(self.Gram_matrix[currIdx].dot(Y*coef_init) + Y.dot(coef_init))
            if currSign <= 0:
                coef_init[currIdx] += 1 
            n_iter += 1
            if self.verbose:
                logger.info(n_iter)
        
        self.coef = coef_init

    
    def predict(self, x):
        return np.sign(x.dot(self.coef) + self.intercept)





    
    
        
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    

iris = datasets.load_iris()

X = iris.data[:,:3]
y = iris.target
X = X[y < 2]
y = y[y < 2]
y = 2*y -1

xxx = perceptron(random_state=3,max_iter=200000,verbose=False, eta=0.01, tol=0)
xxx.fit(X,y)



plt.figure(figsize=(5, 4))
plt.scatter(X[:,0], X[:,1], c=y)
x_vals = np.sort(np.unique(X[:,0]))
y_vals = -xxx.intercept/ xxx.coef[1] - xxx.coef[0] / xxx.coef[1]  * x_vals
plt.plot(x_vals, y_vals, '--') 
plt.show()




from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
fig = plt.figure()

# Add an axes
ax = fig.add_subplot(111,projection='3d')

# plot the surface
xx, yy = np.meshgrid(np.arange(X[:,0].min(), X[:,0].max(), 0.2), np.arange(X[:,1].min(), X[:,1].max(), 0.2))
z = -1. / xxx.coef[2] * (xx * xxx.coef[0] + yy * xxx.coef[1] + xxx.intercept)
ax.plot_wireframe(xx, yy, z, alpha=0.2)

# and plot the point 
ax.scatter(X[:,0] , X[:,1], X[:,2],  c=y)

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
