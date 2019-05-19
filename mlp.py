#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with retropropagation
# learning.
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd


def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
            
        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    def learn(network,samples, epochs=2500, lrate=.1, momentum=0.1):
        # Train 
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n], lrate, momentum )
        # Test
        for i in range(samples.size):
            o = network.propagate_forward( samples['input'][i] )
            print (i, samples['input'][i], '%.2f' % o[0],'(expected %.2f)' % samples['output'][i])
            
        print ("")

    network = MLP(2,2,1)



    #Proyecto 
    # -------------------------------------------------------------------------
    data = pd.read_csv("Absenteeism_at_work.csv")
    #data = pd.read_csv("Wea.csv")
    #df = pd.read_csv("Absenteeism_at_work.csv", usecols = lambda column : column not in ["Age"])
    #print(data.keys())
    #del data['Age']
    targetData = np.array(data[["Absenteeism time in hours"]])
    #print(np.array(targetData)[0])
    data = data.drop('Absenteeism time in hours',axis=1)
    #del data.index.name
    #print(data.get_values()[0])
    tam = len(data.get_values())

    samples = np.zeros(tam, dtype=[('input',  float, 20), ('output', float, 1)])
    #print (tam)
    network.reset()
    for i in range(0, tam):
        #print(data.get_values()[i])
        samples[i][0] = data.get_values()[i]
        samples[i][1] = targetData[i]

    print(samples[736])
    # Example 1 : OR logical function
    # -------------------------------------------------------------------------
    #print ("Learning the OR logical function")
    #network.reset()
    #samples[0] = (0,0), 0
    #print (samples[0])
    #samples[1] = (1,0), 1
    #samples[2] = (0,1), 1
    #samples[3] = (1,1), 1
    #learn(network, samples)

