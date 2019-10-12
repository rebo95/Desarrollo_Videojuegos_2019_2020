import numpy as np
import matplotlib.pyplot as plt
import math
from math import e
from pandas.io.parsers import read_csv

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


def sigmoide_function(X):

    e_z = 1 / np.power(math.e, X) #np.power => First array elements raised to powers from second array, element-wise.

    sigmoide = 1/(1 + e_z)

    print(sigmoide)


def solve_problem():

    X_, Y_ = data_builder(data_csv("ex2data1.csv"))

    data_visualization(X_, Y_)


sigmoide_function([[2,2], [2,2]])