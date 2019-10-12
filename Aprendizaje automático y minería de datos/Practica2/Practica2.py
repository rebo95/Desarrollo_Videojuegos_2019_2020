import numpy as np
import matplotlib.pyplot as plt
import math
from math import e
from pandas.io.parsers import read_csv

import scipy.optimize as opt


def data_csv(file_name):
    "Takes the data from the csv file and tranfers it to a numpy array"

    values_ = read_csv(file_name, header=None).values

    return values_.astype(float)

def data_builder(data):

    X = data[:, :-1]
    Y = data[:, -1]

    return X, Y

def data_visualization(X, Y):

    pos_0 = np.where(Y == 0)
    pos_1 = np.where(Y == 1)

    plt.scatter(X[pos_0, 0], X[pos_0, 1], c = "green")
    plt.scatter(X[pos_1, 0], X[pos_1, 1], marker='+', c = 'k')

    plt.show()

def ones_matrix(X):

    return np.zeros(X.shape) + 1

def sigmoide_function(X):

    e_z = 1 / np.power(math.e, X) #np.power => First array elements raised to powers from second array, element-wise.

    sigmoide = 1/(1 + e_z)

    return sigmoide

def cost(Thetas, X, Y):

    m = X.shape[0] 
    #J(θ) = −(1/m) * (A + B * C)
    #J(θ) = −(1/m) * ((log (g(Xθ)))T * y + (log (1 − g(Xθ)))T * (1 − y))

    #A
    X_Teta = np.dot(X, Thetas)
    g_X_Thetas = sigmoide_function(X_Teta)
    log_g_X_Thetas = np.log(g_X_Thetas)
    T_log_g_X_Thetas = np.transpose(log_g_X_Thetas)
    y_T_log_g_X_Thetas = np.dot(T_log_g_X_Thetas, Y)

    A = y_T_log_g_X_Thetas

    #B
    one_g_X_Thetas = ones_matrix(g_X_Thetas) - g_X_Thetas
    log_one_g_X_Thetas = np.log(one_g_X_Thetas)
    T_log_one_g_X_Thetas = np.transpose(log_one_g_X_Thetas)

    B = T_log_one_g_X_Thetas

    #C
    C = ones_matrix(g_X_Thetas) - Y

    J = (-1/m) * (A + (np.dot(B, C)))


    return J


def gradient(Thetas, X, Y):

    m = X.shape[0]

    X_Teta = np.dot(X, Thetas)
    g_X_Thetas = sigmoide_function(X_Teta)

    X_T = np.transpose(X)

    gradient = (1/m)*(np.dot(X_T, g_X_Thetas - Y ))

    return gradient


def optimized_parameters(Thetas, X, Y):

    result = opt.fmin_tnc(func = cost, x0 = Thetas, fprime = gradient, args = (X, Y) )
    theta_opt = result[0]

    return theta_opt


def solve_problem():

    X_, Y_ = data_builder(data_csv("ex2data1.csv"))
    data_visualization(X_, Y_)

    X_V = X_ #X_ that will be used for use in vectorized methods, with the addition of a collum of ones as the first group of atributes

    X_m = np.shape(X_)[0]

    X_V = np.hstack([np.ones([X_m,1]),X_]) #adding the one collum

    Thetas = np.zeros(X_V.shape[1])

    cost_ = cost(Thetas, X_V, Y_)
    gradient_ = gradient(Thetas, X_V, Y_)

    print(cost_)
    print(gradient_)

    result = optimized_parameters(Thetas, X_V, Y_)
    print(result)

    cost_ = cost(result, X_V, Y_)
    print("Coste optimo ", cost_)






    













def test():
    mat = np.array([[1,math.e,3],[math.e,2,1]])
    Y = np.log(mat)
    print(Y)

solve_problem()


