#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 19:51:34 2018

@author: levipuckett
"""
from Network import Network
from NetDrawClass import NetDraw
import images.MNIST_loader as loader

print ("Let's train a neural network to recognize handwritten digits.")

mini_batch_size = int(input ("Select a mini batch size: "))
learning_rate = float(input ("Select a learning rate: "))
epochs = int(input ("Select number of training epochs: "))
layers = int(input ("Select number of hidden layers in network: "))
sizes = [784]
for i in range(layers):
    sizes.append(int(input("How many nodes in this hidden layer: ")))
sizes.append(10)

print ("Creating network...")
train_images, train_labels, test_images, test_labels = loader.load_pickle()
training_data = list(zip(train_images, train_labels))
test_data = list(zip(test_images, test_labels))
print("Pickles loaded.")
network = Network(sizes)
print ("Network initialized.")
print ()

print ("We now have a network with random assignments for weights and biases.")
print ("It probably won't do so well... Let's see.")
input ("Before we train, let's get a baseline. Here is how the network performs on a set of testing images. (Press Enter).")

network.evaluate(test_data)
print ("Probably not so great. Let's train the network to recognize these digits.")
print ("Watch the weights and biases change as the network is trained.")
input ("Press enter to begin training.")

network.SGD(training_data, mini_batch_size, learning_rate, epochs, updates=True)

print ("Now that the network is trained, let's see how it does with"
       "images it has never seen before!")

wrong = network.evaluate(test_data)

input ("Probably a lot better!\n")

input ("Let's see what it got wrong. Press Enter to scroll through the network's guesses.")

nd = NetDraw()
for image, guess, label in wrong:
    nd.printImage(image, label)
    nd.printNet(network)
    input ("Network thought this was a " + str(guess))
    