import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def carga_csv(file_name):
    "Carga fichero csv especificado y lo devuelve en un array de numpy"

    valores = read_csv(file_name, header=None).values

    return valores.astype(float) #parseamos a float (suponemos que siempre trabajaremos con float)

def hth(x, th): #Hipótesis modelo lineal
    return th[0] + th[1] * x

def H_Theta(X, Z): #Hipótesis del mocelo lineal vectorizada 
    return np.dot(X, Z)

def pinta_costes():


    poblacion = carga_csv('ex1data1.csv')

    X = poblacion[:, :-1]
    Y = poblacion[: , -1]

    X_m = np.shape(X)[0]
    Y_m = np.shape(X)[1]

    X = np.hstack([np.ones([X_m,1]),X])

    num_div = 25
    x_theta0 = np.linspace(-10, 10 ,num_div)
    y_theta1 = np.linspace(-1, 4, num_div)

    xx_thetas0, yy_thetas1 = np.meshgrid(x_theta0, y_theta1)

    thetas = np.zeros((num_div * num_div,2))
    

    print(xx_thetas0)
    print(yy_thetas1)

    dim_0 = xx_thetas0.shape[0]
    dim_1 = xx_thetas0.shape[1]

    J = np.zeros((dim_0, dim_1))


    for i in range(dim_0):
        for j in range(dim_1):

            Z = np.array([xx_thetas0[i,j], yy_thetas1[i,j]])
            J[i,j] = funcion_coste(X, Y, Z)
    
    print(J)

    
    fig = plt.figure()
    ax = Axes3D(fig)

    surf = ax.plot_surface(xx_thetas0, yy_thetas1, J)
    fig.colorbar(surf, shrink = 0.5, aspect = 5)
    plt.show()
    

    






def resuelve_problema():
    poblacion = carga_csv('ex1data1.csv')

    X_ = poblacion[:, :-1]
    Y_ = poblacion[: , -1]

    X_m = np.shape(X_)[0]
    Y_m = np.shape(X_)[1]

    X_ = np.hstack([np.ones([X_m,1]),X_])

    Thetas, Costes = descenso_gradiente(X_, Y_, alpha = 0.01)


def descenso_gradiente(X, Y, alpha):
    
    m = len(X)

    #construimos matriz Z
    th0 = 0.0
    th1 = 0.0

    Z = np.array([th0 ,th1])

    alpha_m = (alpha/m)

    Thetas = np.array([[th0, th1]]) #almacena los thetas que forman parte de la hipotesis h_theta
    Costes = np.array([]) #almacena los costes obtenidos durante el descenso de gradiente

    X_aux = np.array(X[:,1])

    
    for i in range(1500):

        #Calculo de Theta 0
        #Sumatorio para el calculo de Theta0
        sum1 = H_Theta(X, Z) - Y
        sum1_ = sum1.sum()

        #Calculo Theta 1
        #Sumatorio para el calculo de Theta1
        sum2 =  (H_Theta(X, Z) - Y) * X_aux
        sum2_ = sum2.sum()
        
        th0 -= alpha_m * sum1_
        th1 -= alpha_m * sum2_

        Z[0] = th0
        Z[1] = th1

        
        Thetas = np.append(Thetas, [[th0, th1]], axis= 0)

        #funcion de costes
        J = funcion_coste(X,Y, Z)

        Costes = np.append(Costes, [J], axis = 0)



    plt.scatter(X_aux, Y, alpha= 0.5)
    plt.plot([5, 22], [hth(5,Z) , hth(22, Z)], color = "red")
    plt.show()

    return Thetas, Costes

def funcion_coste(X, Y, Theta): #funcion de costes vectorizada
    H = H_Theta(X,Theta)
    Aux = (H-Y)**2
    sumatory = Aux.sum()/(2 * len(X))
    return sumatory


pinta_costes()
#resuelve_problema()

