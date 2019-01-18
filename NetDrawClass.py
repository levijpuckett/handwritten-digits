#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 18:58:36 2018

@author: levipuckett
"""

#drawing library.
import pygame as p
from pygame import draw, display
import numpy as np

import images.MNIST_loader as MNIST_loader

#constants for drawing.
SIZE = [1200,700]
START_OF_NUM = [0,210]
START_OF_NET = [300,20]
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREY = (128,128,128)

class NetDraw:
    def __init__(self):
        p.init() 
        self.screen = display.set_mode(SIZE)
        display.set_caption('Ready to draw numbers')
        self.screen.fill(WHITE)
        display.flip()
        for event in p.event.get():
            if event.type == p.QUIT:
                pass
    
    def printImage(self, image, label):                
        display.set_caption("Current number is a " + str(label))
        y = START_OF_NUM[1]
        for i in range(28):
            x = START_OF_NUM[0]
            y += 10
            for j in range(28):
                draw.rect(self.screen, [image[28*i + j]*255,image[28*i + j]*255,image[28*i + j]*255],[x,y,10,10])
                x += 10
        display.flip()
        for event in p.event.get():
            pass  
    
    def printNet(self, network):
        x = START_OF_NET[0]
        last_layer = []
        this_layer = []
        index = 0
        draw.ellipse(self.screen, GREY, [x,START_OF_NET[1] + 105,50,500])
        this_layer.append( (START_OF_NET[0] + 25,350) )
        x += int( SIZE[0] / network.number_of_layers ) 
        w_maxes = []
        b_maxes = []
        for wm in network.weights:
            w_maxes.append(np.max(np.abs(wm)))
        for bm in network.biases:
            b_maxes.append(np.max(np.abs(bm)))
        w_adj = 255 / max(w_maxes)
        b_adj = 255 / max(b_maxes)
            
        for size, biases in zip(network.sizes[1:], network.biases):
            last_layer = this_layer.copy()
            this_layer.clear()
            y = START_OF_NET[1]
            radius = 10
            y_step = int( SIZE[1] / size )
            for n, bias in zip(range(size), biases):
                if bias < 0:
                    draw.circle(self.screen, (-b_adj*bias,0,0), (x,y), radius)
                else:
                    draw.circle(self.screen, (0,b_adj*bias,0)
                    ,(x,y), radius)
                this_layer.append( (x,y) )
                y += y_step
            x += int( (SIZE[0] - START_OF_NET[0]) / network.number_of_layers )
            if last_layer and this_layer:
                weights_matr = network.weights[index]
                index+=1
                for right, weights in zip(this_layer, weights_matr):
                    for left, weight in zip(last_layer, weights):
                        if weight < 0:
                            draw.line(self.screen, (-w_adj*weight,0,0), left, right)
                        else:
                            draw.line(self.screen, (0,w_adj*weight,0), left, right)
        display.flip()
        for event in p.event.get():
            pass 

if __name__ == "__main__":
    from Network import Network
    network = NetDraw()
    train_images, train_labels, test_images, test_labels = MNIST_loader.load_pickle()
    
    net = Network([784,16,16,10])
    network.printNet(net)
    
    for image, label in zip(train_images,train_labels):
        network.printImage(image, label)
