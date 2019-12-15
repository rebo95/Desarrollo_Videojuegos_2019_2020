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

import random

#___________________________________________________________________________________________________________________

def debugInitializeWeights(fan_in, fan_out):
    """
    Initializes the weights of a layer with fan_in incoming connections and
    fan_out outgoing connections using a fixed set of values.
    """

    # Set W to zero matrix
    W = np.zeros((fan_out, fan_in + 1))

    # Initialize W using "sin". This ensures that W is always of the same
    # values and will be useful in debugging.
    W = np.array([np.sin(w) for w in range(np.size(W))]).reshape((np.size(W, 0), np.size(W, 1)))

    return W


def computeNumericalGradient(J, theta):
    """
    Computes the gradient of J around theta using finite differences and
    yields a numerical estimate of the gradient.
    """

    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4

    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * tol)
        perturb[p] = 0

    return numgrad


def checkNNGradients(costNN, reg_param):
    """
    Creates a small neural network to check the back propogation gradients.
    Outputs the analytical gradients produced by the back prop code and the
    numerical gradients computed using the computeNumericalGradient function.
    These should result in very similar values.
    """
    # Set up small NN
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate some random test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to get random X
    X = debugInitializeWeights(input_layer_size - 1, m)

    # Set each element of y to be in [0,num_labels]
    y = [(i % num_labels) for i in range(m)]

    ys = np.zeros((m, num_labels))
    for i in range(m):
        ys[i, y[i]] = 1

    # Unroll parameters
    nn_params = np.append(Theta1, Theta2).reshape(-1)

    # Compute Cost
    cost, grad = costNN(nn_params, input_layer_size, hidden_layer_size, 
    num_labels, X, ys, reg_param)

    def reduced_cost_func(p):
        """ Cheaply decorated nnCostFunction """
        return costNN(p, input_layer_size, hidden_layer_size, num_labels, 
        X, ys, reg_param)[0]

    numgrad = computeNumericalGradient(reduced_cost_func, nn_params)

    # Check two gradients
    np.testing.assert_almost_equal(grad, numgrad)
    return (grad - numgrad)

#___________________________________________________________________________________________________________________

def load_data(file_name):
    data = loadmat(file_name)

    y = data['y']
    y_2 = np.ravel(y)
    X = data['X']

    return  X, y_2

def load_wwights_neuronal_red(file_name): #Carga los parámetros de una red neuronal
    weights = loadmat(file_name)
    theta1 , theta2 = weights ['Theta1'] , weights ['Theta2']

    return theta1, theta2

def rnd_selection_data(X, num_samples): #Elige aleatoriamente una selección de entradas dentro de la población de datos
    sample = np.random.choice(X.shape[0], num_samples)
    rnd_selected_rows_matrix = X[sample, :]

    return rnd_selected_rows_matrix

#___________________________________________________________________________________________________________________

def data_printer(X, num_samples):
    rnd_samples_matrix = rnd_selection_data(X, num_samples)
    displayData(rnd_samples_matrix)
    plt.show()

def displayData(X):#Convierte los valores de entrada en un elemento representable pro una imagen
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

def displayImage(im):#Nos premite imprimir una imagen
    fig2, ax2 = plt.subplots()
    image = im.reshape(20, 20).T
    ax2.imshow(image, cmap='gray')
    return (fig2, ax2)
#___________________________________________________________________________________________________________________

def unrollVect(a, b): #nos permite desplegar en un vector otros dos
    thetaVec_ = np.concatenate((np.ravel(a), np.ravel(b)))
    return thetaVec_

def rollVector(params, num_entradas, num_ocultas, num_etiquetas):
    #pliega el vector params en dos vectores de parámetros correspondinetes a los vectores de pesos de cada una de las capas de nuestra red
    vector1 = np.matrix(np.reshape(params[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1))))
    vector2 = np.matrix(np.reshape(params[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1))))

    return vector1, vector2

def generate_Random_Weights(L_in, L_out): #Genera un array de pesos para una capa de una red neuronal con entrada L_in y salida L_out

    e_ini = math.sqrt(6)/math.sqrt(L_in + L_out)

    e_ini= 0.12

    weights = np.zeros((L_out, 1 + L_in))

    for i in range(L_out):
        for j in range(1 + L_in):

            rnd = random.uniform(-e_ini, e_ini)
            weights[i,j] = rnd

    return weights


def y_onehot(y, X, num_etiquetas):
    #Devuelve la salida en forma de matriz lista para ser utilizada por nuestros métodos de la red neuronal
    m = X.shape[0]

    y = (y - 1)
    y_onehot = np.zeros((m, num_etiquetas))  # 5000 x 10
    
    for i in range(m):
        y_onehot[i][y[i]] = 1
    
    return y_onehot

#___________________________________________________________________________________________________________________

def vectors_coincidence_percentage(a, b):
    #Calcula el porcentaje de coincidencia dados dos vectores a, b
    coincidences_array = a == b

    coincidences = sum(map(lambda coincidences_array : coincidences_array == True, coincidences_array  ))
    percentage =100 * coincidences/coincidences_array.shape

    return percentage

#___________________________________________________________________________________________________________________

def sigmoid(x):
    return 1/(1 + np.exp((-x)))

def sigmoid_Derived(x):
    return x * (1 - x)

def sigmoid_Gradient(z):
    sig_z = sigmoid(z)
    return np.multiply(sig_z, (1 - sig_z))

def forward(X, theta1, theta2):#Método pasada hacia adelante para la implementación de la red neuronal
    #Nos devuelve los parámetros de activación de la red neuronal y el valor h
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T 
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1) 
    z3 = a2 * theta2.T 
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


def cost(params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje):
#Funcion que calcula el coste base(sin regularizar)
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y) 

    theta1 = np.matrix(np.reshape(params[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1))))
    theta2 = np.matrix(np.reshape(params[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1))))

    h = forward(X, theta1, theta2)[4]

    J = (np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))).sum() / m

    return J, theta1, theta2

def cost_Regularized(params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje):
#Cálculo del coste con el ajuste de regularización
    m = X.shape[0]

    J_, theta1, theta2 = cost(params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje)
    
    J_regularized =  J_ + (float(tasa_aprendizaje) /
            (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    
    return J_regularized


def backProp_Deltas(a1, z2, a2, z3, h, theta1, theta2, y, m):
#Calculo de los gradientes
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)
    
    d3 = h - y

    z2 = np.insert(z2, 0, values=np.ones(1), axis=1)

    d2 = np.multiply((theta2.T * d3.T).T, sigmoid_Gradient(z2))

    delta1 += (d2[:, 1:]).T * a1
    delta2 += d3.T * a2

    delta1 = delta1 / m
    delta2 = delta2 / m

    return delta1, delta2


def backProp_Deltas_regularized(a1, z2, a2, z3, h, theta1, theta2, y, m, tasa_aprendizaje):
#Calculo de los gradientes en formato regularizado 
    delta1, delta2 = backProp_Deltas(a1, z2, a2, z3, h, theta1, theta2, y, m)

    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * tasa_aprendizaje) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * tasa_aprendizaje) / m

    return delta1, delta2


def backprop(params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje, regularize = True):
#Pasada hacia adelante y hacia atrás en nuestra red neuronal, nos calcula el gradiente y el coste correspondientes a nuestra red neuronal
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    theta1 = np.matrix(np.reshape(params[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1))))
    theta2 = np.matrix(np.reshape(params[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1))))


    a1, z2, a2, z3, h = forward(X, theta1, theta2)

    if regularize:
        delta1, delta2 = backProp_Deltas_regularized(a1, z2, a2, z3, h, theta1, theta2, y, m, tasa_aprendizaje)
    else :
        delta1, delta2 = backProp_Deltas(a1, z2, a2, z3, h, theta1, theta2, y, m)

    J = cost_Regularized(params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje)

    grad = unrollVect(delta1, delta2)

    return J, grad

def minimice(backprop, params, num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje):
    #Calcula los parámetros optimos de pesos para nuestra red neuronal
    fmin = opt.minimize(fun=backprop, x0=params, args=(num_entradas, num_ocultas, num_etiquetas, X, y, tasa_aprendizaje), 
    method='TNC', jac=True, options={'maxiter': 70})

    result = fmin.x
    return result

#___________________________________________________________________________________________________________________

def neuronal_prediction_vector(sigmoids_matrix) :
    #calcula los valores predichos por la red neuronal dada una matriz de sigmoides 
    # Será generada por la funcón forward.

    samples = sigmoids_matrix.shape[0]
    y = np.zeros(samples)
    
    for i in range(samples):
        y[i] = np.argmax(sigmoids_matrix[i, :]) +1

    return y

def neuronal_succes_percentage(X, y, weights1, weights2) :
    #Compara los valores predichos por la red neuronal para unos 
    sigmoids_matrix = forward(X, weights1, weights2)[4]
    y_ = neuronal_prediction_vector(sigmoids_matrix)
    percentage = vectors_coincidence_percentage(y_, y)

    return percentage

#___________________________________________________________________________________________________________________

#np.set_printoptions(threshold=sys.maxsize)

def main():

    X, y = load_data("ex4data1.mat")
    
    tasa_aprendizaje = 1
    num_etiquetas = 10 #num_etiquetas = num_salidas
    num_entradas = 400
    num_ocultas = 25

    theta1 = generate_Random_Weights(num_entradas, num_ocultas)
    theta2 = generate_Random_Weights(num_ocultas, num_etiquetas)

    params_rn = unrollVect(theta1, theta2)
    y_ = y_onehot(y, X, num_etiquetas)

    params_optimiced = minimice(backprop, params_rn, num_entradas, num_ocultas, num_etiquetas, X, y_, tasa_aprendizaje)

    theta1_optimiced, theta2_optimiced = rollVector(params_optimiced, num_entradas, num_ocultas, num_etiquetas)

    percentage = neuronal_succes_percentage(X, y, theta1_optimiced, theta2_optimiced)
    
    print("Percentage neuronal red : ", percentage)


main()