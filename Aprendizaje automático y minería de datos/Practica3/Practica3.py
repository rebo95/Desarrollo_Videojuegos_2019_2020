import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

import math
from math import e
from pandas.io.parsers import read_csv

import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures


def load_data(file_name):
    data = loadmat(file_name)

    y = data['y']
    X = data['X']

    return y, X

def draw_rnd_selection_data(X):
    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()

def two_power(X):
    return X**2

def H(X, Z): #Hipótesis del mocelo lineal vectorizada 
    return np.dot(X, Z)
    
def g_z(X):

    e_z = 1 / np.power(math.e, X) #np.power => First array elements raised to powers from second array, element-wise.

    sigmoide = 1/(1 + e_z)

    return sigmoide

def cost(Thetas, X, Y):

    m = X.shape[0] 
    #J(θ) = −(1/m) * (A + B * C)
    #J(θ) = −(1/m) * ((log (g(Xθ)))T * y + (log (1 − g(Xθ)))T * (1 − y))

    #A
    X_Teta = np.dot(X, Thetas)
    g_X_Thetas = g_z(X_Teta)
    log_g_X_Thetas = np.log(g_X_Thetas)
    T_log_g_X_Thetas = np.transpose(log_g_X_Thetas)
    y_T_log_g_X_Thetas = np.dot(T_log_g_X_Thetas, Y)

    A = y_T_log_g_X_Thetas

    #B
    one_g_X_Thetas = 1 - g_X_Thetas
    log_one_g_X_Thetas = np.log(one_g_X_Thetas)
    T_log_one_g_X_Thetas = np.transpose(log_one_g_X_Thetas)

    B = T_log_one_g_X_Thetas

    #C
    C = 1 - Y

    J = (-1/m) * (A + (np.dot(B, C)))


    return J


def cost_regularized(Thetas, X, Y, h):

    m = X.shape[0] 
    Thetas_ = Thetas

    #J(θ) = (cost(Thetas, X, Y)) + D
    #J(θ) = [−(1/m) * ((log (g(Xθ)))T * y + (log (1 − g(Xθ)))T * (1 − y)))] + (λ/2m)*E(Theta^2)

    cost_ = cost(Thetas, X, Y)
   
    #D
    Thetas_ = two_power(Thetas_)

    D = h/(2*m) * np.sum(Thetas_)

    J_regularized = (cost_) + D

    return J_regularized


def gradient(Thetas, X, Y):

    m = X.shape[0]
    #(δJ(θ)/δθj) =(1/m)*XT*(g(Xθ) − y)
    

    X_Teta = np.dot(X, Thetas)
    g_X_Thetas = g_z(X_Teta)

    X_T = np.transpose(X)

    gradient = (1/m)*(np.dot(X_T, g_X_Thetas - Y ))

    return gradient

def gradient_regularized(Thetas, X, Y, h):

    m = X.shape[0]
    #(δJ(θ)/δθj) =(1/m)*XT*(g(Xθ) − y) + (λ/2m)(Theta)
    gradient_ = gradient(Thetas, X, Y)

    g_regularized = gradient_ + (h/m)*Thetas
    
    return g_regularized


def optimized_parameters_regularized(Thetas, X, Y, reg):

    result = opt.fmin_tnc(func = cost_regularized, x0 = Thetas, fprime = gradient_regularized, args = (X, Y, reg) )
    theta_opt = result[0]

    return result


def oneVsAll(X, y, num_etiquetas, reg):

    #Theta = np.zeros([num_etiquetas ,X.shape[1]]) #contains the thetas of each case or clase, row 0 have the values for the first number, row two for the second ...
    Theta = np.zeros(X.shape[1])
    optimized_parameters_regularized(Theta, X, y, 1)

    return Theta

def main():

    reg = 1

    y, X = load_data("ex3data1")
    draw_rnd_selection_data(X)

    Thetas = oneVsAll(X_poly, y, 10, reg)

    print(Thetas.shape)
    


main()