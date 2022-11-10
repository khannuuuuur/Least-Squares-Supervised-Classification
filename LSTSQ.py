###########################################
#         ECE 174 MINI PROJECT 1
###########################################
# AUTHOR: Conner Hsu
# PID: A16665092
###########################################
import os.path
from pathlib import Path
import sys

import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import math

def saveNp(path_name, arr):
    if '/' not in path_name:
        np.save(path_name, arr)
        return
    lastSlash = path_name.rindex('/')
    path = path_name[0:lastSlash+1]
    name = path_name[lastSlash:]
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path_name, arr)

class FeatureMap:
    # Initializes the W matrix and b vector
    def initWb(d, L, scale=1, u=0):
        folder = 'Wb_' + str(scale) + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        ls = os.listdir(folder)
        if len(ls) == 0:
            FeatureMap.W = np.random.normal(loc=u, scale=scale, size=(d, L))
            saveNp(folder + 'W' + str(d) + '_L' + str(L) + '.npy', FeatureMap.W)
            FeatureMap.b = np.random.normal(loc=u,scale=scale, size=(1, L))
            saveNp(folder + 'b' + str(d) + '_L' + str(L) + '.npy', FeatureMap.b)
            return

        # if files exist in Wb/ , assume they are in the form
        # b{d}_L{L}.npy or W{d}_L{L}.npy
        for file in ls:
            if file[0] == 'W':
                FeatureMap.W = np.load(folder + file)
            if file[0] == 'b':
                FeatureMap.b = np.load(folder + file)

        dfile = FeatureMap.W.shape[0]
        Lfile = FeatureMap.W.shape[1]

        # If W and b are not big enough, just expand the existing W and b.
        if Lfile < L or dfile < d:
            for file in ls:
                os.remove(folder + file)
            newW = np.random.normal(loc=u,scale=scale, size=(d, L))
            newW[0:dfile, 0:Lfile] = FeatureMap.W
            FeatureMap.W = newW
            saveNp(folder + 'W' + str(d) + '_L' + str(L) + '.npy', FeatureMap.W)

            newb = np.random.normal(loc=u,scale=scale, size=(1, L))
            newb[0, 0:Lfile] = FeatureMap.b
            FeatureMap.b = newb
            saveNp(folder + 'b' + str(d) + '_L' + str(L) + '.npy', FeatureMap.b)

    def __init__(self, d, g, L=1000):
        FeatureMap.initWb(d, L)
        self.W = FeatureMap.W[0:d, 0:L]
        self.b = FeatureMap.b[0, 0:L]

        self.L = L
        self.g = g
        self.d = d

        self.name = g.__name__ + '_L' + str(self.L) + '_' + str(d)

    def applyMap(self, dataX):
        if len(dataX.shape) != 2:
            dataX = dataX.reshape(1, dataX.shape[0])

        B = np.tile(self.b, (dataX.shape[0], 1))

        result = np.matmul(dataX, self.W)+B
        result = self.g(result)

        return result

    def sigmoid(arr):
        return 1 / (1 + np.exp(-arr))
    def identity(arr):
        return arr
    def sinusoid(arr):
        return np.sin(np.radians(arr))
    def ReLU(arr):
        return np.clip(arr, 0, np.max(arr))

# Builds a matrix that processes the output of a 1v1 classifier.
# Returns a 45x10 matrix that will count all the "votes" for each 1v1 classifier
def get_counter(n=10):
    counter = np.zeros((n*(n-1)//2, n))
    k=0
    for i in range(n):
        for j in range(i):
            counter[k, i] = 1
            counter[k, j] = -1
            k+=1
    return counter

def get_weights1v1(trainX, trainY, h=None):
    path = 'weights1v1/'
    hname = ''
    if h is not None:
        hname=h.name
    name = 'weights1v1_' + hname + '.npy'

    # Try loading prexisting weights if they already exist
    if Path(path+name).is_file():
        return np.load(path+name)

    if h is not None:
        trainX = h.applyMap(trainX)
    trainX = np.append(trainX, np.ones((trainX.shape[0],1)), axis=1) # Add bias

    weights = []
    n = 10
    for i in range(n):
        trainYi = (trainY == i).astype('int')

        for j in range(i):
            trainYij = trainYi - (trainY==j).astype('int')

            # Delete rows that don't correspond with i or j
            zeros = np.where(trainYij==0)
            trainYij = np.delete(trainYij, zeros)
            trainXij = np.delete(trainX, zeros, axis=0)

            pinvXij = np.linalg.pinv(trainXij)
            weightsij = np.matmul(pinvXij, trainYij)
            weights.append(weightsij)

    weights = np.array(weights).T
    saveNp(path+name, weights)
    return weights
def get_weights1vAll(trainX, trainY, h=None):
    path = 'weights1vAll/'
    hname = ''
    if h is not None:
        hname=h.name
    name = 'weights1vAll_' + hname + '.npy'

    # Try loading prexisting weights if they already exist
    if Path(path+name).is_file():
        return np.load(path+name)

    n = 10
    trainYlabeled = []
    for i in range(n):
        trainYlabeled.append((2*(trainY==i)-1)[0]) # Normalize Y to 1 or -1
    trainY =  np.array(trainYlabeled).T

    if h is not None:
        trainX = h.applyMap(trainX)
    trainX = np.append(trainX, np.ones((trainX.shape[0],1)), axis=1) # Add bias
    pinvX = np.linalg.pinv(trainX)
    weights = np.matmul(pinvX, trainY)

    saveNp(path+name, weights)
    return weights
def get_testX(testX, h=None, new=False, write_new=True):
    path = 'testX/'

    hname = ''
    if h is not None:
        hname=h.name

    name = 'testX_' + hname+'.npy'
    if not new and Path(path+name).is_file():
        return np.load(path+name)

    if h is not None:
        testX = h.applyMap(testX)
    testX = np.append(testX, np.ones((testX.shape[0],1)), axis=1) # Add bias

    if write_new:
        saveNp(path+name, testX)
    return testX


def test(testX, testY, weights, counter=None, print=True):

    results = np.matmul(testX, weights)

    # For 1v1 classifier
    if counter is not None:
        results = 2*(results > 0).astype('int')-1 #normalize to -1 or 1
        results = np.matmul(results, counter)

    answer = np.argmax(results, axis=1)

    # Determine confusion matrix
    confusionMatrix = np.zeros((10, 10))
    for i in range(0, 10):
        for j in range(0, 10):
            confusionMatrix[i,j] = np.dot((testY==i).astype('int'), (answer==j).astype('int'))
    confusionMatrix = confusionMatrix.astype('int')
    if print:
        printConfusionMatrixLatex(confusionMatrix)
    return (1-np.trace(confusionMatrix)/np.sum(confusionMatrix))*100
def printConfusionMatrixLatex(A):
    for i in range(A.shape[0]):
        print(i, end='   & ')
        for j in range(A.shape[1]):
            print('{:5d}'.format(A[i,j]), end=' &')
        print('{:5d}'.format(np.sum(A[i])), '& ',end='')
        print('{:5.2f}'.format((1-A[i,i]/np.sum(A[i]))*100), '\\% \\\\')
    print('\\midrule\nAll', end=' & ')
    for i in range(A.shape[1]):
        print('{:5d}'.format(np.sum(A[:,i])), end=' &')
    print('{:5d}'.format(np.sum(A)), '&', '{:5.2f}'.format((1-np.trace(A)/np.sum(A))*100), '\% \\\\')

def problem2(mnist):

    trainX = mnist['trainX']/255
    trainY = mnist['trainY']
    testX = mnist['testX']/255
    testY = mnist['testY']

    counter=get_counter()

    testX = get_testX(testX)

    print('1vAll')
    weights = get_weights1vAll(trainX, trainY)
    test(testX, testY, weights)

    print('1v1')
    weights2 = get_weights1v1(trainX, trainY)
    test(testX, testY, weights2, counter=counter)
def problem3_1(mnist):

    trainX = mnist['trainX']/255
    trainY = mnist['trainY']
    testX = mnist['testX']/255
    testY = mnist['testY']

    counter=get_counter()

    l = 40
    functions = [FeatureMap.identity, FeatureMap.sigmoid, FeatureMap.sinusoid, FeatureMap.ReLU]
    for func in functions:
        print('='*l + '\n' + ' '*((l-len(func.__name__))//2) + func.__name__ + '\n' + '='*l)

        h = FeatureMap(trainX.shape[1], func)
        testXH = get_testX(testX, h=h)

        print('1vAll')
        weights = get_weights1vAll(trainX, trainY, h=h)
        test(testXH, testY, weights, print=True)

        print('1v1')
        weights = get_weights1v1(trainX, trainY, h=h)
        test(testXH, testY, weights, counter=counter,print=True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def problem3_2(mnist):
    trainX = mnist['trainX']/255
    trainY = mnist['trainY']
    testX = mnist['testX']/255
    testY = mnist['testY']

    functions = [FeatureMap.identity,FeatureMap.sigmoid, FeatureMap.sinusoid, FeatureMap.ReLU]
    #functions = [FeatureMap.identity, FeatureMap.sinusoid]
    errorrate = {}
    for func in functions:
        errorrate[func] = []

    Lrange = np.linspace(100, 2000, 20)
    for L in Lrange:
        for func in functions:
            h = FeatureMap(trainX.shape[1], func, L=int(L))
            testXH = get_testX(testX, h=h)

            weights = get_weights1vAll(trainX, trainY, h=h)
            errorrate[func].append(test(testXH, testY, weights, print=False))


    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots()
    line, = ax.plot(Lrange, errorrate[FeatureMap.identity], marker='x')
    line.set_label(FeatureMap.identity.__name__)
    for func in functions[1:]:
        line, = ax.plot(Lrange, errorrate[func], marker='+')
        line.set_label(func.__name__)
    ax.grid()
    ax.legend()
    ax.set_xlabel('$L$', fontsize=12)
    ax.set_ylabel('Error \%', fontsize=12)
    ax.set_title('Error rate of 1vAll classifiers', fontsize=12)
    plt.show()
def problem3_3(mnist):
    trainX = mnist['trainX']/255
    trainY = mnist['trainY']
    testX = mnist['testX']/255
    testY = mnist['testY']

    functions = [FeatureMap.identity, FeatureMap.sigmoid, FeatureMap.sinusoid, FeatureMap.ReLU]
    #functions = [FeatureMap.sigmoid, FeatureMap.ReLU]
    #functions = [FeatureMap.identity, FeatureMap.sinusoid]
    weights = {}
    errorrate = {}
    hs = {}
    for func in functions:
        hs[func] = FeatureMap(trainX.shape[1], func)
        weights[func] = get_weights1vAll(trainX, trainY, h=hs[func])
        errorrate[func] = []

    errorlevels = np.linspace(0, 250, 100)
    noise = np.random.normal(size=(1, testX.shape[1]))
    if Path('noise.npy').is_file():
        noise = np.load('noise.npy')
    saveNp('noise', noise)
    noisenorm = np.linalg.norm(noise)
    for errorlevel in errorlevels:
        for func in functions:
            testXH = get_testX(testX+noise*errorlevel/noisenorm, h=hs[func], new=True, write_new=False)
            errorrate[func].append(test(testXH, testY, weights[func], print=False))

    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(1)
    for func in functions:
        line, = ax.plot(errorlevels, errorrate[func])
        line.set_label(func.__name__)
    ax.legend()
    ax.set_xlabel('Error level, $\epsilon$', fontsize=12)
    ax.set_ylabel('Error \%', fontsize=12)
    ax.set(ylim=(0, 100))
    plt.yticks(np.arange(0, 100, 10))
    plt.grid()
    ax.set_title('Error rate of 1vAll classifier with different feature mappings', fontsize=12)
    plt.show()
    # Identity breakpoint: 2
    # sinusoid breakpoint: 7
    # sigmoid breakpoint: 66
    # relu breakpoint: 75
# Tests hand drawn images in folder called images/
def test_images(mnist):

    trainX = mnist['trainX']/255
    trainY = mnist['trainY']

    h = FeatureMap(trainX.shape[1], FeatureMap.ReLU, L=2000)
    weights = get_weights1v1(trainX, trainY, h=h)
    counter=get_counter()

    constrast = 2
    for i in range(1, 17):
        image = np.array(Image.open('images/' + str(i)+'.png'))
        image = image.reshape(28,28)
        image = np.clip(constrast*(image.astype('float32')-128)+128, 0, 255).astype('uint8')

        plt.imshow(image)
        plt.show()

        test = image.reshape(784)/255
        if h is not None:
            test = h.applyMap(test)
        test = np.append(test, 1)
        result = np.matmul(test, weights)
        if counter is not None:
            result = 2*(result > 0).astype('int')-1
            result = np.matmul(result, counter)

        answer = np.argmax(result)
        print(answer, '{:.2f}'.format(result[answer]))


def main():

    mnist = sio.loadmat("mnist.mat")

    #problem2(mnist)
    problem3_1(mnist)
    #problem3_2(mnist)
    #problem3_3(mnist)

    #best1v1(mnist)
    #test_images(mnist)

if __name__ == '__main__':
    main()
