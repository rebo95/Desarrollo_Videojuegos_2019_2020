import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    "Carga fichero csv especificado y lo devuelve en un array de numpy"

    valores = read_csv(file_name, header=None).values

    return valores.astype(float) #parseamos a float (suponemos que siempre trabajaremos con float)

def hth(x, th):
    return th[0] + th[1] * x

def H_Theta(X, Z):
    return np.dot(X, Z)


def resuelve_problema():
    poblacion = carga_csv('ex1data1.csv')

    X_ = poblacion[:, :-1]
    Y_ = poblacion[: , -1]

    X_m = np.shape(X_)[0]
    Y_m = np.shape(X_)[1]

    X_ = np.hstack([np.ones([X_m,1]),X_])

    descenso_gradiente(X_, Y_, alpha = 0.01)


def descenso_gradiente(X, Y, alpha):
    
    

    m = len(X)

    #construimos matriz Z
    th0 = 0.0
    th1 = 0.0
    Z = np.array([th0 ,th1])

    alpha_m = (alpha/m)
    Thetas = np.array([[th0, th1]])
    Costes = np.array([])

    X_aux = np.array(X[:,1])

    for i in range(1500):

        sum1 = H_Theta(X, Z) - Y
        sum1_ = sum1.sum()

        
        sum2 =  (H_Theta(X, Z) - Y) * X_aux
        sum2_ = sum2.sum()
        
        th0 -= alpha_m * sum1_
        th1 -= alpha_m * sum2_

        Z[0] = th0
        Z[1] = th1
        print("separado ", Z)



        Thetas = np.append(Thetas, [[th0, th1]], axis= 0)

        H = H_Theta(X,Z)
        Aux = (H-Y)**2
        sum3 = Aux.sum()/(2 * len(X))
        Costes = np.append(Costes, [sum3], axis = 0)


    plt.scatter(X_aux, Y, alpha= 0.5)
    plt.plot([5, 22], [hth(5,Z) , hth(22, Z)], color = "red")
    plt.show()







resuelve_problema()

