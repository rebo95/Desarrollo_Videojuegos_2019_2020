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

def load_data(file_name):
    data = loadmat(file_name)

    y = data['y']
    y_2 = np.ravel(y)
    X = data['X']

    return  X, y_2

def load_wwights_neuronal_red(file_name):
    weights = loadmat(file_name)
    theta1 , theta2 = weights ['Theta1'] , weights ['Theta2']

    return theta1, theta2

    
def draw_rnd_selection_data(X):
    sample = np.random.choice(X.shape[0], 100)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()


def main():
    X, Y = load_data("ex4data1.mat")
    theta1, theta2 = load_wwights_neuronal_red("ex4weights.mat")
    draw_rnd_selection_data(X)


main()