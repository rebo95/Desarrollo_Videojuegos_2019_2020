import numpy as np
import random
from scipy.integrate import quad
import math as mt


#Implementación integral iterativa

def sin_function(x):
    return np.sin(x)

def x2_function(x):
    y = x*2
    return y


def functon_max(fun, x_values_array):

    y_value  = fun(x_values_array[0])
    max_value =  y_value

    for i in range(len(x_values_array)):
        y_value = fun(x_values_array[i])
        if  y_value > max_value:
            max_value = y_value
    
    return max_value

def x_values(a,b, num_points):

    length = b - a

    if length < 1:
        num_of_divisions = num_points
    else:
        num_of_divisions = num_points * length

    x_values_array =  np.linspace(a, b, num_of_divisions)

    return x_values_array

def area_calculator(a,b,num_points,points_inside_area,function_maximum):
    integral = (points_inside_area/num_points)*(b-a)*function_maximum
    return integral

def points_behind_function_area(fun, a, b, num_points, function_maximum):

    points_inside_area = 0

    for j in range(num_points):

        x_value_area_point = np.random.uniform(a,b)
        y_value_area_point = np.random.uniform(0,function_maximum)

        if fun(x_value_area_point) > y_value_area_point:
            points_inside_area += 1


    return points_inside_area

def integra_mc(fun, a, b, num_points = 10000):

    x_values_array = x_values(a, b, num_points)

    max_value = functon_max(fun, x_values_array)

    points_inside_area = points_behind_function_area(fun, a, b, num_points, max_value)
    
    print(area_calculator(a, b, num_points, points_inside_area, max_value))


#Implementación cálculo de área usando operaciones entre vectores 
def integra_mc_vectorial(fun, a ,b , num_points = 10000):

    x_values_array = x_values(a, b, num_points)
    y_values_function = fun(x_values_array)

    max_value = np.amax(y_values_function)

    x_random_values = np.random.uniform(a, b, num_points)
    y_random_values = np.random.uniform(0, max_value, num_points)

    y_random_values_function = fun(x_random_values)

    elements_within_area = y_random_values[y_random_values < y_random_values_function] #el vector resultante contiene tantos elementos como elementos del array sobre el que se aplica cumplan la condicion dada
    num_of_points_behind_fun = len(elements_within_area)


    print(area_calculator(a, b, num_points, num_of_points_behind_fun, max_value))
    






integra_mc_vectorial(sin_function, 0, 2)
integra_mc(sin_function, 0,2)


integral_python_method = quad(sin_function, 0,2)
print(integral_python_method)