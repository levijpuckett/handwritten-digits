#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 19:12:04 2018

@author: levipuckett
"""
import numpy as np
import random
random.seed()

import images.MNIST_loader as loader
from NetDrawClass import NetDraw

class Network(object):
    '''Implements a Network object.
    Constructor: takes a list whose length determines the number
    of layers in the network, and each entry indicates the number
    of nodes in the layer.
    Eg., sizes = [2,4,2] results in a network with 3 layers: one with 2 nodes,
    one with 4 nodes, and one with 2 nodes.
    '''
    def __init__(self, sizes):
        self.number_of_layers = len(sizes)
        self.sizes = sizes
        
        #Build weights array.
        #List filled with 2D Numpy arrays.
        #Each entry in the list is a layer.
        #Each row indicates node in current layer,
        #each column indicates link to previous layer's node.
        #so self.weights[0][2,3] is layer 1: connection between
        #layer 1, node 2 and layer 0 (input), node 3.
        self.weights = [ np.random.randn(x,y)
                        for x,y in zip(sizes[1:],sizes[:-1]) ]

        #Build weights array.
        #list filled with Numpy vectors.
        #Each entry in the list is a layer.
        #each component of the vector is the 
        #bias of that node in the layer.
        #self.biases[2][1] is the bias of
        #node 1 in layer 2.
        self.biases = [ np.random.randn(x)
                        for x in sizes[1:]]
    
    def feedforward(self, image):
        '''computes the output of the network by feeding forward the image,
        storing activations and z vectors along the way.
        Returns a 2-tuple of (activations,zs).'''
        activation = image #store activation of current layer.
        activations = [image] #list to store all activations of all layers.
        zs = []# list to store 'unsquished' activations, layer by layer.
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activations, zs
    
    def SGD(self, training_data, mini_batch_size, learning_rate, 
            epochs, updates=False):
        '''Trains the network using SGD.
        training_data is a list of two-tuples containing 
        an image and a label.'''
        n = len(training_data)
        
        if updates:
            print ('Begining training.')
            print ('mini batch size:', mini_batch_size)
            print ('learning rate:', learning_rate)
            print ('epochs:', epochs)
                   
        for i in range(epochs):
            if updates:
                print('Begining epoch', i)
                nd.printNet(self)
            #Break training data into mini batches and shuffle randomly.
            random.shuffle(training_data)
            mini_batches = [ training_data[j : j + mini_batch_size]
                            for j in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                #compute delta_b and delta_a, the averaged error in weights
                #and biases, respectively.
                delta_b = [np.zeros_like(b) for b in self.biases]
                delta_w = [np.zeros_like(w) for w in self.weights]
                for x,y_int in mini_batch:
                    b_err, w_err = self.backprop(x,y_int)
                    
                    #add these errors to the running total.
                    delta_b = [db + be for db,be in zip(delta_b, b_err)]
                    delta_w = [dw + we for dw,we in zip(delta_w, w_err)]
                
                #apply the averaged gradient to the weights.
                self.biases = [b - (learning_rate / len(mini_batch)) * 
                               db for b,db in zip(self.biases, delta_b)]
                self.weights = [w - (learning_rate / len(mini_batch)) * 
                                dw for w,dw in zip(self.weights, delta_w)]
        if updates:
            print ('\nFinished training.')
            
    
    def backprop(self, x, y_int):
        '''computes and returns the error in the network.
        "error" -> gradient of cost function.
        
        x is numpy array (784,) image.
        y_int is integer expected output of network.
        
        returns list of numpy arrays that contain the derivative
        of the cost function:
        b_err -> error in the biases.
        w_err -> error in the weights.
        '''
        #make y_int expected output as numpy vector.
        y = np.zeros(self.sizes[-1])
        y[y_int] = 1.0
        
        err_b = [np.zeros_like(b) for b in self.biases]
        err_w = [np.zeros_like(w) for w in self.weights]
        
        #feedfoward.
        activations, zs = self.feedforward(x)
        
        #find error in output layer:
        error = (activations[-1] - y) * sigmoid_prime(zs[-1])
        
        err_b[-1] = error
        err_w[-1] = np.dot(error.reshape(len(error),1), 
             np.transpose(activations[-2].reshape(len(activations[-2]),1)))
        #backward pass through the network
        #error in each layer is:
        #   error = weights(layer + 1) * error(layer + 1)  * sigmoid_prime(z)
        #z and weighst are from current layer.
        #layer + 1 refers to the layer in front of the current layer.
        #this is what makes it a backward pass.
        #eg, for the last hidden layer (hl):
        #error = weights(out_layer) * error(out_layer) * sigmoid_prime(z)
        #Note: * implies Hadamard product.
        for layer in range(2, self.number_of_layers):
            #Each increment of the layer variable refers to a step
            #backwards in the network (negative indexing).
            #starting at the first hidden layer (the output layer error is
            #already stored in delta).
            z = zs[-layer]
            w = np.transpose(self.weights[-layer + 1])
            sp = sigmoid_prime(z) #gather sigmoid_prime for error.
            error = np.dot(w, error) * sp
            
            err_b[-layer] = error
            
            a_in = activations[-layer - 1].reshape(len(activations[-layer-1]),1)
            err_w[-layer] = np.dot(error.reshape(len(error),1), np.transpose(a_in))

        return (err_b, err_w)
    
    def compute_output(self, image):
        '''like feedforward, computes the output of the network.
        Only returns the final layer activation, used for evaluating
        network performance.'''
        activation = image #store activation of current layer.
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,activation) + b
            activation = sigmoid(z)
        return activation
    
    def evaluate(self, test_data):
        correct = 0
        n = len(test_data)
        wrong_guesses = []
        for image, label in test_data:
            output = self.compute_output(image)
            if np.argmax(output) == label:
                correct += 1
            else:
                wrong_guesses.append((image, np.argmax(output), label))
        print ('network guessed', correct, 'out of', n, 'images.')
        print ('That is %.2f percent!' %((correct / n) * 100))
        return wrong_guesses
            
        
### Math functions ###
def sigmoid(z):
    '''Returns a 'squished' z vector according to:
        sig = 1 / (1 + e^(-z))
        returns as numpy array.'''
    return 1.0 / (1.0 + np.exp(-z))
            
def sigmoid_prime(z):
    '''Returns the derivative of sigmoid function wrt z'''
    return sigmoid(z) * (1.0 - sigmoid(z))
            
nd = NetDraw()
if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = loader.load_pickle()
    print('pickles loaded.')
    training_data = list(zip(train_images, train_labels))
    test_data = list(zip(test_images, test_labels))
    sizes = [784,16,10]
    net = Network(sizes)
    
    nd.printNet(net)
    #evaluate untrained, random network.
    net.evaluate(test_data)
    #train the network.
    net.SGD(training_data, 10, 3.0, 30, updates=True)
    #evaluate trained network.
    wrong = net.evaluate(test_data)
    
    nd.printNet(net)