import numpy as np
import matplotlib.pyplot as plt
import math
from math import e
from pandas.io.parsers import read_csv

import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

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

def draw_frontier(X, Y, Thetas):

    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = sigmoide_function(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(Thetas))
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')

    data_visualization(X, Y)

def draw_frontier_3D(X, Y, Thetas):

    
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = sigmoide_function(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(Thetas))
    h = h.reshape(xx1.shape)

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(xx1, xx2, h, cmap= cm.coolwarm, linewidths= 0, antialiaseds = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    plt.show()



def draw_frontier_regularized(X, Y, Theta, poly):

    plt.figure()
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = sigmoide_function(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(Theta))
    h = h.reshape(xx1.shape)
    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')

    data_visualization(X, Y)

def draw_frontier_regularized_3D(X, Y, Theta, poly):

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    h = sigmoide_function(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(Theta))
    h = h.reshape(xx1.shape)

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(xx1, xx2, h, cmap= cm.coolwarm, linewidths= 0, antialiaseds = False)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    plt.show()


    

def H(X, Z): #Hipótesis del mocelo lineal vectorizada 
    return np.dot(X, Z)

def two_power(X):
    return X**2

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
    g_X_Thetas = sigmoide_function(X_Teta)

    X_T = np.transpose(X)

    gradient = (1/m)*(np.dot(X_T, g_X_Thetas - Y ))

    return gradient

def gradient_regularized(Thetas, X, Y, h):

    m = X.shape[0]
    #(δJ(θ)/δθj) =(1/m)*XT*(g(Xθ) − y) + (λ/2m)(Theta)
    gradient_ = gradient(Thetas, X, Y)

    g_regularized = gradient_ + (h/m)*Thetas
    
    return g_regularized


def optimized_parameters(Thetas, X, Y):

    result = opt.fmin_tnc(func = cost, x0 = Thetas, fprime = gradient, args = (X, Y) )
    theta_opt = result[0]

    return theta_opt

def optimized_parameters_regularized(Thetas, X, Y, h):

    result = opt.fmin_tnc(func = cost_regularized, x0 = Thetas, fprime = gradient_regularized, args = (X, Y, h) )
    theta_opt = result[0]

    return theta_opt

def logistic_regresion_evaluation(X, Y, Z):

    H_ = H(X, Z)
    H_sigmoid = sigmoide_function(H_)

    H_sigmoid_evaluated = (H_sigmoid >= 0.5).astype(np.float) #every value that keeps the condition will return true. astyping it to int it will turn it into a one value, having an array ready to compare with Y_
    
    comparison_array = H_sigmoid_evaluated == Y #returns an array where each value will be true if the condition is keeped

    coincidences = comparison_array[comparison_array == True] #return an array with only the elements that keep the condition from the original array

    percentage_correct_clasification = (coincidences.shape[0] / comparison_array.shape[0]) * 100.0 #how many trues(coincidences) do we have in comparison with all the succesful and not succesful coincidences (trues/(trues+falses))

    return(percentage_correct_clasification)
    

def logistic_regresion():

    X_, Y_ = data_builder(data_csv("ex2data1.csv"))
    

    X_V = X_ #X_ that will be used for use in vectorized methods, with the addition of a collum of ones as the first group of atributes

    X_m = np.shape(X_)[0]

    X_V = np.hstack([np.ones([X_m,1]),X_]) #adding the one collum

    Thetas = np.zeros(X_V.shape[1])

    cost_ = cost(Thetas, X_V, Y_)
    gradient_ = gradient(Thetas, X_V, Y_)

    #testing and comparing if the resoults are correct having in mind the resoults given in the assignment document
    print(cost_)
    print(gradient_)

    
    optimized_thetas = optimized_parameters(Thetas, X_V, Y_)#Optimus cost is obtained by calling the cost ecuation using the optimized thetas


    print("Thetas optimos: ", optimized_thetas)
    cost_ = cost(optimized_thetas, X_V, Y_)
    print("Coste optimo : ", cost_)


    percentage_correct_clasification = logistic_regresion_evaluation(X_V, Y_, optimized_thetas)

    print("Porcentaje de acierto : ", percentage_correct_clasification)

    draw_frontier(X_, Y_, optimized_thetas )
    draw_frontier_3D(X_, Y_, optimized_thetas )
    
    

def regularized_logistic_regresion(h = 1):

    X_, Y_ = data_builder(data_csv("ex2data2.csv"))
    
    
    poly = PolynomialFeatures(degree=6)
    X_poly = poly.fit_transform(X_)

    #theta and alpha inicialization
    Thetas = np.zeros(X_poly.shape[1])

    cost_r = cost_regularized(Thetas, X_poly, Y_, h)
    gradient_r = gradient_regularized(Thetas, X_poly, Y_, h)
    print(cost_r)
    print(gradient_r)

    optimized_thetas_regularized = optimized_parameters_regularized(Thetas, X_poly, Y_, h)

    draw_frontier_regularized(X_, Y_, optimized_thetas_regularized, poly )
    draw_frontier_regularized_3D(X_, Y_, optimized_thetas_regularized, poly )
    


def test():
    mat = np.array([[1,2,3],[3,2,1]])
    mat = two_power(mat)
    print(mat)

logistic_regresion()
regularized_logistic_regresion()


