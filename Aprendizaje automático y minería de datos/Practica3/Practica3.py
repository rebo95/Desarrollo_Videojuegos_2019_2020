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

np.set_printoptions(threshold=sys.maxsize)
def hipothesis_value(X, y, num_etiquetas, reg, Thetas_matrix):

    training_sample_dimension = X.shape[0]//num_etiquetas

    step = 0

    X_ = X[step:training_sample_dimension, :]
    y_ = y[step:training_sample_dimension ]

    '''
    for i in range(num_etiquetas):
        step = training_sample_dimension * i
        end_step =step + training_sample_dimension

        X_ = X[step:end_step, :]
        y_ = y[step:end_step ]
        Z_ = Thetas_matrix[i, :]

        H_ = H(X_, Z_)
        H_sigmoid = g_z(H_)

        ////////////////////////
        Z_ = Thetas_matrix[9, :]
        H_ = H(X, Z_)
        H_sigmoid = g_z(H_)
        H_sigmoid_evaluated = (H_sigmoid >= 0.5).astype(np.float)
    '''

    for j in range(num_etiquetas): #selects the sample to compare or analize within all the training samples 

        step = training_sample_dimension * j
        end_step = step + training_sample_dimension
        X_ = X[step:end_step, :]

        for i in range(num_etiquetas): #selects the theta optimized values for each tag or numner, the coincidences or number ones will be greater in the elements that fits with the
                                        #same number the thetas values represent 
            Z_ = Thetas_matrix[i, :]

            H_ = H(X_, Z_)
            H_sigmoid = g_z(H_)

            H_sigmoid_evaluated = (H_sigmoid >= 0.5).astype(np.float)

            ones = sum(map(lambda k : k == True, H_sigmoid_evaluated)) #count the number of elemnets that match the condition in k within the array H_sigmoid_evaluated

            probability = 100 * ones/H_sigmoid_evaluated.shape

            print("Number : ",j ,"Test_value = ", i, "% = ", probability)
    


def main():

    reg = 1

    y, X = load_data("ex3data1")
    #draw_rnd_selection_data(X)

    X_ones = np.hstack([np.ones([X.shape[0],1]),X]) #adding the one collum
    Thetas = np.zeros(X_ones.shape[1])

    Thetas_matrix = oneVsAll(X_ones, y, 10, reg, Thetas)

    hipothesis_value(X_ones, y, 10, 1, Thetas_matrix)



    
    


main()