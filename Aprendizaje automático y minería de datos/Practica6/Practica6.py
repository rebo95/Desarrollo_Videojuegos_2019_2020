import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from sklearn import svm 

#_________________________________________________________________________________________
def load_data(file_name):
    data = loadmat(file_name)

    X = data['X']
    y = data['y']
    y_r = np.ravel(y)

    return X, y, y_r

def displayData(X, y):

    pos_0 = np.where(y == 0)
    neg_0 = np.where(y == 1)

    plt.plot(X[:,0][pos_0], X[:,1][pos_0], "yo")
    plt.plot(X[:,0][neg_0], X[:,1][neg_0], "k+")

#_________________________________________________________________________________________

def draw_Linear_KernerFrontier(X, y, svm_function):

    w = svm_function.coef_[0]
    a = -w[0] / w[1]

    #seleccionamos dos puntos de la recta para representarla
    p1 = np.array([X[:,0].min(), X[:,0].max()])
    p2 = a * p1 - (svm_function.intercept_[0]) / w[1]

    #Frontera de separación
    plt.plot(p1, p2, c = 'y')
    displayData(X, y)
    plt.show()

def draw_Non_Linear_KernelFrontier(X, y , model, sigma):
   
    #Datos que conformarán la curva que servirá de frontera
    x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
    x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = model.predict(gaussian_Kernel(this_X, X, sigma))

    #Frontera de separación
    plt.contour(X1, X2, vals, colors="y", linewidths = 0.1 )
    
    displayData(X, y)
    plt.show()
#_________________________________________________________________________________________

def SVM_linear_training(X, y, c_param):
    svm_ = svm.SVC( kernel = "linear", C = c_param)
    svm_.fit (X, y)
    return svm_

def SVM_gaussian_training(X, y, c_param, tol, max_i, sigma):

    svm_ = svm.SVC(C = c_param, kernel="precomputed", tol = tol, max_iter = max_i)
    return svm_.fit(gaussian_Kernel(X, X, sigma=sigma), y)
#_________________________________________________________________________________________

def gaussian_Kernel(X1, X2, sigma):
    Gram = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.ravel()
            x2 = x2.ravel()
            Gram[i, j] = np.exp(-np.sum(np.square(x1 - x2)) / (2 * (sigma**2)))
    return Gram
#_________________________________________________________________________________________

def part1_main():
    #Parte 1.1
    X, y, y_r = load_data("ex6data1.mat")
    # C = 1
    c_param = 1
    svm_function = SVM_linear_training(X, y_r, c_param)
    draw_Linear_KernerFrontier(X, y_r, svm_function)

    # C = 100
    c_param = 100
    svm_function = SVM_linear_training(X, y_r, c_param)
    draw_Linear_KernerFrontier(X, y_r, svm_function)

def part2_main():
    #Parte 1.2
    c_param = 1
    sigma = 0.1
    tool = 1e-3
    iterations = 100
    X1, y1, y1_r = load_data("ex6data2.mat")
    svm_function_n_l = SVM_gaussian_training(X1, y1_r, c_param, tool, iterations, sigma)
    draw_Non_Linear_KernelFrontier(X1, y1_r, svm_function_n_l, sigma)
    
def main():
    part1_main()
    part2_main()
#_________________________________________________________________________________________



main()