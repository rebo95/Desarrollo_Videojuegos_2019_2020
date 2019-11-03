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


def backprop(params_rn , num_entradas , num_ocultas , num_etiquetas , X, y, reg):

    theta1 = np. reshape(params_rn [: num_ocultas * (num_entradas + 1) ] , (num_ocultas , (num_entradas + 1)))
    theta2 = np. reshape(params_rn[num_ocultas * (num_entradas + 1) :] , (num_etiquetas , (num_ocultas + 1)))
    print("Hello world")


def main():
    X, Y = load_data("ex4data1.mat")
    theta1, theta2 = load_wwights_neuronal_red("ex4weights.mat")

    data_printer(X, 100)




    



main()