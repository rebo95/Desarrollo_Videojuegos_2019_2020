import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

import math
from math import e
from pandas.io.parsers import read_csv

import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

import sys 

from IPython.display import display
import pandas as pd

def rnd_selection_data(X, num_samples):
    sample = np.random.choice(X.shape[0], num_samples) #it returns the index of the random selected rows
    rnd_selected_rows_matrix = X[sample, :]

    return rnd_selected_rows_matrix


def displayData(X):
    num_plots = int(np.size(X, 0)**.5)
    fig, ax = plt.subplots(num_plots, num_plots, sharex=True, sharey=True)
    plt.subplots_adjust(left=0, wspace=0, hspace=0)
    img_num = 0
    for i in range(num_plots):
        for j in range(num_plots):
            # Convert column vector into 20x20 pixel matrix
            # transpose
            img = X[img_num, :].reshape(20, 20).T
            ax[i][j].imshow(img, cmap='Greys')
            ax[i][j].set_axis_off()
            img_num += 1

    return (fig, ax)


def displayImage(im):
    fig2, ax2 = plt.subplots()
    image = im.reshape(20, 20).T
    ax2.imshow(image, cmap='gray')
    return (fig2, ax2)

def data_printer(X, num_samples):
    rnd_samples_matrix = rnd_selection_data(X, num_samples)
    displayData(rnd_samples_matrix)
    plt.show()

def load_data(file_name):
    data = loadmat(file_name)

    y = data['y']
    y_2 = np.ravel(y)
    X = data['X']

    return  X, y_2

def load_wwights_neuronal_red(file_name): #parameters of a neruonal red
    weights = loadmat(file_name)
    theta1 , theta2 = weights ['Theta1'] , weights ['Theta2']

    
    return theta1, theta2

def sigmoid(X):

    e_z = 1 / np.power(math.e, X) #np.power => First array elements raised to powers from second array, element-wise.

    sigmoide = 1/(1 + e_z)

    return sigmoide


np.set_printoptions(threshold=sys.maxsize)

def cost(theta1, theta2, X, y):

    m = X.shape[0] 
    h = H_f_p(X, theta1, theta2)
    print(theta1.shape)
    num_labels = 10

    y = (y - 1)
    y_onehot = np.zeros((m, num_labels))  # 5000 x 10
    
    for i in range(m):
        y_onehot[i][y[i]] = 1
        
    
    J = 0

    for i in range(m):
        for j in range(10):
            first_term = np.multiply(-y_onehot[i, j], np.log(h[i, j]))
            second_term = np.multiply((1 - y_onehot[i, j]), np.log(1 - h[i, j]))
            J += np.sum(first_term - second_term)
    J = J / m
    

    #J = np.sum(np.multiply(-y_onehot, np.log(h)) - np.multiply((1 - y_onehot), np.log(1 - h))) / m #using np.multiply because it is multiplication member to member

    return J


def cost_regularized(theta1, theta2, X, y, h):

    m = X.shape[0]


    #Not vectorized
    '''
    j_1 = theta1.shape[0]
    k_1 = theta1.shape[1]

    j_2 = theta2.shape[0]
    k_2 = theta2.shape[1]

    first_term = 0
    for j in range(j_1):
        for k in range(k_1):
            first_term += np.power(theta1[j,k], 2)

    second_term = 0
    for j in range(j_2):
        for k in range(k_2):
            second_term += np.power(theta2[j,k], 2)
    '''

    #Vectorized

    first_term = np.sum(np.power(theta1, 2))
    second_term = np.sum(np.power(theta2, 2))

    r_term = (h/(2*m)) * (first_term + second_term)

    J = cost(theta1, theta2, X, y) + r_term

    return J




def gradient(Thetas, X, Y):

    m = X.shape[0]
    #(δJ(θ)/δθj) =(1/m)*XT*(g(Xθ) − y)
    

def gradient_regularized(Thetas, X, Y, h):

    m = X.shape[0]
    #(δJ(θ)/δθj) =(1/m)*XT*(g(Xθ) − y) + (λ/2m)(Theta)


def optimized_parameters_regularized(Thetas, X, Y, reg):

    result = opt.fmin_tnc(func = cost_regularized, x0 = Thetas, fprime = gradient_regularized, args = (X, Y, reg) )
    theta_opt = result[0]

    return theta_opt


def oneVsAll(X, y, num_etiquetas, reg, Thetas):

    y_= (y == 10).astype(np.int)
    Thetas_ = optimized_parameters_regularized(Thetas, X, y_, reg)

    optimiced_parameters_matrix = Thetas_

    for i in range(1, num_etiquetas):
        y_ = (y == i).astype(np.int)
        Thetas_ = optimized_parameters_regularized(Thetas, X, y_, reg)
        optimiced_parameters_matrix = np.vstack((optimiced_parameters_matrix, Thetas_))
    
    
    return optimiced_parameters_matrix


def H_f_p(X, theta1, theta2):
        
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    return h

def backprop(params_rn , num_entradas , num_ocultas , num_etiquetas , X, y, reg):

    theta1 = np.reshape(params_rn[: num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1) :] , (num_etiquetas , (num_ocultas + 1)))


def main():

    X, y = load_data("ex4data1.mat")
    
    theta1, theta2 = load_wwights_neuronal_red("ex4weights.mat")

    

    reg = 1
    num_etiquetas = 10 #num_etiquetas = num_salidas
    num_entradas = 400
    num_ocultas = 25

    params_rn = (np.random.random(size = num_ocultas * (num_entradas + 1) + num_etiquetas * (num_ocultas + 1)) - 0.5) * 0.25

    J_ = cost_regularized(theta1, theta2, X, y, 1)
    
    print(J_)








    



main()