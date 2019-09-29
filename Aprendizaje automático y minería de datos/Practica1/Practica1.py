import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    "Carga fichero csv especificado y lo devuelve en un array de numpy"

    valores = read_csv(file_name, header=None).values

    return valores.astype(float) #parseamos a float (suponemos que siempre trabajaremos con float)


def pinta_puntos():

    poblacion = carga_csv('ex1data1.csv')

    X = np.array(poblacion[:,0]) #sería nuestra variable independiente
    Y = np.array(poblacion[:,1]) #sería nuestra variable independiente o el valor resultante de evaluar la función en X

    plt.scatter(X,Y, alpha= 0.5)
    plt.show()

pinta_puntos()