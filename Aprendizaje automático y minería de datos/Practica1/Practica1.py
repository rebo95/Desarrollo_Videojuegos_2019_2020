import numpy as numpy
from pandas.io.parsers import read_csv

def carga_csv(file_name):
    "Carga fichero csv especificado y lo devuelve en un array de numpy"

    valores = read_csv(file_name, header=None).values

    return valores.astype(float) #parseamos a float (suponemos que siempre trabajaremos con float)
