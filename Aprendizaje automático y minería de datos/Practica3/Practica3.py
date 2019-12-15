import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

import math
from math import e
from pandas.io.parsers import read_csv

import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

import sys 



def load_data(file_name):
    data = loadmat(file_name)

    y = data['y']
    y_2 = np.ravel(y)
    X = data['X']

    return y_2, X

def load_data_neuronal_red(file_name):
    weights = loadmat(file_name)
    theta1 , theta2 = weights [ 'Theta1' ] , weights [ 'Theta2' ]

    return theta1, theta2

def draw_rnd_selection_data(X):
    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
    plt.show()

def two_power(X):
    return X**2

def H(X, Z): #Hipótesis del mocelo lineal vectorizada 
    return np.dot(X, Z)
    
def sigmoid(X):

    e_z = 1 / np.power(math.e, X) #np.power => First array elements raised to powers from second array, element-wise.

    sigmoide = 1/(1 + e_z)

    return sigmoide

def vectors_coincidence_percentage(a, b):
    
    coincidences_array = a == b

    coincidences = sum(map(lambda coincidences_array : coincidences_array == True, coincidences_array  ))
    percentage =100 * coincidences/coincidences_array.shape

    return percentage

def cost(Thetas, X, Y):

    m = X.shape[0] 
    #J(θ) = −(1/m) * (A + B * C)
    #J(θ) = −(1/m) * ((log (g(Xθ)))T * y + (log (1 − g(Xθ)))T * (1 − y))

    #A
    X_Teta = np.dot(X, Thetas)
    g_X_Thetas = sigmoid(X_Teta)
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
    g_X_Thetas = sigmoid(X_Teta)

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
    #TENGO QUE IR TRANSFORMANDO LA REGRESION LOGÍSTICA PARA LOS VALORES DE Y DE TAL MANERA QUE HAY 1 DONDE COINCIDE CON EL VALOR DEL NUMERO REPRESENTADO Y 0 EN EL RESTO 
    #PARA EL VALOR DE 2 POR EJEMPLO TENDRIAMOS UNA Y DE [0 0 1 0 0 0 0 0 0 0 ]
    #PARA EL VALOR DE 5 POR EJEMPLO TENDRIAMOS UNA Y DE [0 0 0 0 0 1 0 0 0 0 ]
    #SIGMOIDE MAXIMO ES EL QUE ESTÁ MAS SEGURO DE LA SALIDA LO QUE IMPLICA QUE SE REFIERE A ESE NUMERO NESIMO CON LOS VALORES DE PESOS DE LA MATRIZ


def Sample_clasifier(sample, num_etiquetas, Thetas_matrix):

    sigmoids = np.zeros(num_etiquetas)

    for i in range(num_etiquetas): #selects the theta optimized values for each tag or numner, the coincidences or number ones will be greater in the elements that fits with the
                                        #same number the thetas values represent 
        Z_ = Thetas_matrix[i, :]

        H_ = H(sample, Z_)
        H_sigmoid = sigmoid(H_)
        sigmoids[i] = H_sigmoid 

    num_tag = np.argmax(sigmoids)

    return num_tag

def all_samples_comparator_percentage(X, y, num_etiquetas, reg, Thetas):

    Thetas_matrix = oneVsAll(X, y, num_etiquetas, reg, Thetas)

    samples = X.shape[0]
    y_ = np.zeros(samples)
    y = np.where(y == 10, 0, y)

    for i in range(samples):
        y_[i] = Sample_clasifier(X[i, :], num_etiquetas, Thetas_matrix)

    percentage = vectors_coincidence_percentage(y_, y)

    return percentage

def forward_propagation(X, theta1, theta2):
    #V1
    '''
    a1 = X
    a1_ones = np.hstack([np.ones([a1.shape[0],1]),a1])
    z2 = H(a1_ones, np.transpose(theta1))
    a2 = g_z(z2)

    a2_ones = np.hstack([np.ones([a2.shape[0],1]),a2])

    z3 = H(a2_ones, np.transpose(theta2))
    a3 = g_z(z3)

    y = a3
    '''
    
    m = X.shape[0]
    a1 = np.hstack([np.ones([m, 1]), X])
    z2 = np.dot(a1, theta1.T)
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)

    return h


def neuronal_prediction_vector(sigmoids_matrix) :
    samples = sigmoids_matrix.shape[0]
    y = np.zeros(samples)
      
    for i in range(samples):
        y[i] = np.argmax(sigmoids_matrix[i, :]) +1 
    return y

def neuronal_succes_percentage(X, y, theta1, theta2) :

    sigmoids_matrix = forward_propagation(X, theta1, theta2)
    y_ = neuronal_prediction_vector(sigmoids_matrix)
    percentage = vectors_coincidence_percentage(y_, y)

    return percentage


np.set_printoptions(threshold=sys.maxsize)
def main():

    reg = 1
    num_etiquetas = 10
    y, X = load_data("ex3data1")
    #draw_rnd_selection_data(X)


    X_ones = np.hstack([np.ones([X.shape[0],1]),X]) #adding the one collum

    
    Thetas = np.zeros(X_ones.shape[1])

    percentage = all_samples_comparator_percentage(X_ones, y, num_etiquetas, reg, Thetas)
    
    print("Percentage one vs all : ", percentage)


    theta1, theta2 = load_data_neuronal_red("ex3weights.mat")
    
    percentage = neuronal_succes_percentage(X, y, theta1, theta2)
    
    print("Percentage neuronal red : ", percentage)

main()