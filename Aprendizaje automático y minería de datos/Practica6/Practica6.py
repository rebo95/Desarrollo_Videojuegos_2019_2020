import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from sklearn import svm 

def load_data(file_name):
    data = loadmat(file_name)

    X = data['X']
    y = data['y']
    y_r = np.ravel(y)

    return X, y_r

def displayData(X, y):

    pos_0 = np.where(y == 0)
    neg_0 = np.where(y == 1)


    plt.plot(X[:,0][pos_0], X[:,1][pos_0], "yo")
    plt.plot(X[:,0][neg_0], X[:,1][neg_0], "k+")
    plt.show()

def SVM_linear_function(X, y, c_param):

    svm_ = svm.SVC( kernel = "linear", C = c_param)
    svm_.fit (X, y)
    return svm_

def drawLinearKernerFrontier(X, y, svm_function):

    w = svm_function.coef_[0]
    a = -w[0] / w[1]

    xx = np.array([X[:,0].min(), X[:,0].max()])
    yy = a * xx - (svm_function.intercept_[0]) / w[1]

    #Frontera de separación
    plt.plot(xx, yy, c = 'y')
    displayData(X, y)


def main():
    X, y = load_data("ex6data1.mat")

    c_param = 1
    svm_function = SVM_linear_function(X, y, c_param)
    drawLinearKernerFrontier(X, y, svm_function)

    c_param = 100
    svm_function = SVM_linear_function(X, y, c_param)
    drawLinearKernerFrontier(X, y, svm_function)



main()
