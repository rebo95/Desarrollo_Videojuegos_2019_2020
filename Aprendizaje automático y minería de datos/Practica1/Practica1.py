import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    "Carga fichero csv especificado y lo devuelve en un array de numpy"

    valores = read_csv(file_name, header=None).values

    return valores.astype(float) #parseamos a float (suponemos que siempre trabajaremos con float)

def hth(x, th):
    return th[0] + th[1] * x



def one_variable_linear_regresion(alpha_ = 0.01, num_iter = 1500):

    poblacion = carga_csv('ex1data1.csv')

    X = np.array(poblacion[:,0]) #sería nuestra variable independiente
    Y = np.array(poblacion[:,1]) #sería nuestra variable independiente o el valor resultante de evaluar la función en X

    m = len(X)
    
    #construimos matriz Z
    th0 = 0
    th1 = 0
    Z = [th0 ,th1]

    alpha_m = (alpha_/m)
    

    for i in range(1500):

        sumatory = 0
        sumatory1 = 0

        for j in range(m):
            sumatory += hth(X[j], Z) - Y[j]
            sumatory1 += (hth(X[j], Z) - Y[j]) * X[j]

        th0 -= alpha_m * sumatory
        th1 -= alpha_m * sumatory1 


        sumatory_j = 0
        for k in range(m):
             sumatory_j += (hth(X[k], Z) - Y[k]) ** 2

        J = sumatory_j/(2*m)

        #Reconstruimos Z a partir de los nuevos valores de 
        Z = [th0, th1]

        print(J)

    #pintado de la gráfica y nue de puntos
    plt.scatter(X, Y, alpha= 0.5)
    plt.plot([5, 22], [hth(5,Z) , hth(22, Z)], color = "red")

    X_axis_th0 = np.linspace(-10, 10, 20)
    Y_axis_th1 = np.linspace(-1, 4, 20)



    plt.show()
    

def multiple_variable_linear_regresion(alpha_ = 0.01, num_iter = 1500):
        poblacion_multiple_variables = carga_csv("ex1data2.csv")

        pies_cuadrados = np.array(poblacion_multiple_variables[:, 0])
        num_habitaciones = np.array(poblacion_multiple_variables[:, 1])
        precio = np.array(poblacion_multiple_variables[:, 2])

        media_pies = np.average(pies_cuadrados)
        media_num_habit = np.average(num_habitaciones)
        media_precio = np.average(precio)

        #para la desviacion estandar usar npy std









#one_variable_linear_regresion()
multiple_variable_linear_regresion()