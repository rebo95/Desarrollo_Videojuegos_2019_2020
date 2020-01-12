import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import codecs
from sklearn import svm 

import re
import nltk
import nltk.stem.porter

import os

#_________________________________________________________________________________________
#Procesado de emails
#_________________________________________________________________________________________

def preProcess(email):
    
    hdrstart = email.find("\n\n")
    if hdrstart != -1:
        email = email[hdrstart:]

    email = email.lower()
    # Strip html tags. replace with a space
    email = re.sub('<[^<>]+>', ' ', email)
    # Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)
    # Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    # The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email)
    return email


def email2TokenList(raw_email):
    """
    Function that takes in a raw email, preprocesses it, tokenizes it,
    stems each word, and returns a list of tokens in the e-mail
    """

    stemmer = nltk.stem.porter.PorterStemmer()
    email = preProcess(raw_email)

    # Split the e-mail into individual words (tokens) 
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]',
                      email)

    # Loop over each token and use a stemmer to shorten it
    tokenlist = []
    for token in tokens:

        token = re.sub('[^a-zA-Z0-9]', '', token)
        stemmed = stemmer.stem(token)
        #Throw out empty tokens
        if not len(token):
            continue
        # Store a list of all unique stemmed words
        tokenlist.append(stemmed)

    return tokenlist

#_________________________________________________________________________________________
#Procesado diccionario
#_________________________________________________________________________________________

def getVocabDict(reverse=False):
    """
    Function to read in the supplied vocab list text file into a dictionary.
    Dictionary key is the stemmed word, value is the index in the text file
    If "reverse", the keys and values are switched.
    """
    vocab_dict = {}
    with open("vocab.txt") as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key

    return vocab_dict

#_________________________________________________________________________________________
#Procesado de archivos de texto
#_________________________________________________________________________________________

def procces_txt_data(path):

    file = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                file.append(os.path.join(r, file))

    j = 0
    vocab_dict = keyDictionaryLoader()

    arrayDict = np.array(list(vocab_dict.items()))

    X = np.zeros((len(file), len(arrayDict)))

    for f in file:
        email_contents = codecs.open(f, "r", encoding="utf−8", errors="ignore").read()

        email = email2TokenList(email_contents)

        aux = np.zeros(len(arrayDict))

        for i in range(len(email)):
            index = np.where(arrayDict[:, 0] == email[i])
            aux[index] = 1

        X[j] = aux
        j = j + 1

    print("Archivos de ", path, "leídos y guardados en X.")
    return X

#_________________________________________________________________________________________


def load_data(file_name):
    data = loadmat(file_name)

    X = data['X']
    y = data['y']
    y_r = np.ravel(y)

    return X, y, y_r

def load_data_validation(file_name):
    data = loadmat(file_name)
    X = data['X']
    y = data['y']
    y_r = np.ravel(y)

    Xval = data['Xval']
    yval = data['yval']

    return X, y_r , Xval, yval


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
def optimal_C_sigma_Parameters(X, y_r, Xval, yval, max_i, tool ):
    
    predictions = dict() #almacenaremos la infrmacion relevante en un diccionario
    for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for sigma in[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
            model = SVM_gaussian_training(X, y_r, C, tool, max_i, sigma )
            prediction = model.predict(gaussian_Kernel( Xval, X, sigma))
            predictions[(C, sigma)] = np.mean((prediction != yval).astype(int))
            
    
    C, sigma = min(predictions, key=predictions.get)
    return C, sigma
#_________________________________________________________________________________________
#Procesado de datos e-mail

def load_corpus(file_name):
    email_contents = open(file_name, 'r', encoding='utf-8', errors = 'ignore').read()
    email = email2TokenList(email_contents)
    return email

def keyDictionaryLoader():
    return getVocabDict()

#_________________________________________________________________________________________
#Practica6 Parte1
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

def part3_main():
    #Parte 1.3

    tool = 1e-3
    iterations = 100

    X, y_r , Xval, yval = load_data_validation("ex6data3.mat")

    optC, optSigma = optimal_C_sigma_Parameters(X, y_r, Xval, yval, iterations, tool)

    svm_function_optimal_C_sigma = SVM_gaussian_training(X, y_r, optC, tool, iterations, optSigma)
    draw_Non_Linear_KernelFrontier(X, y_r, svm_function_optimal_C_sigma, optSigma)
    
#_________________________________________________________________________________________
#Practica6 Parte2
#_________________________________________________________________________________________
    

def mainPart1():
    #part1_main()
    #part2_main()
    #part3_main()
#_________________________________________________________________________________________

def mainPart2():
    
    print("Hola mundo")





def main():
    mainPart1()


main()