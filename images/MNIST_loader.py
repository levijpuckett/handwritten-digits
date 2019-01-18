#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:32:53 2018

@author: levipuckett
"""

import numpy as np
import os.path

labelsPath = 'imagestrain-labels.idx1-ubyte'
imagesPath = 'train-images.idx3-ubyte'

def from_bytes(bytez):
    return int.from_bytes(bytez, byteorder='big', signed=False)

def load_images():
    '''load_images returns a tuple (images, labels).
    images -> numpy array of size (60000,784)
    labels -> numpy array of size (60000)
    ordered.'''
    with open(imagesPath, 'rb') as file:
        from_bytes(file.read(4)) #get rid of the magic number.
        num = from_bytes(file.read(4)) #number of images.
        rows = from_bytes(file.read(4)) #number of rows.
        cols = from_bytes(file.read(4)) #number of columns.
        
        images = np.empty((num,rows * cols))
        
        for image in range(num):
            #update on progress.
            print ('\r%.2f percent of images loaded.' % (image / num * 100.0), end='')
            #get pixels for image.
            for pixel in range(rows * cols):   
                images[image, pixel] = ( from_bytes(file.read(1)) / 255.0 )
        file.close()
    print ()
    with open(labelsPath, "rb") as file:
        file.read(4)
        num = from_bytes(file.read(4))
        labels = np.empty(num, dtype = int)
        for i in range(num):
            print ('\r%.2f percent of labels loaded.' % (i / num * 100.0), end='')
            labels[i] = from_bytes(file.read(1))
        file.close()
    print ()
    
    return images, labels

def make_pickle():
    labelsPath = 'images/train-labels.idx1-ubyte'
    imagesPath = 'images/train-images.idx3-ubyte'

    train_images, train_labels = load_images()

    np.save('training_images', train_images)
    np.save('training_labels', train_labels)
    
    labelsPath = 'images/t10k-labels.idx3-ubyte'
    imagesPath = 'images/t10k-images.idx3-ubyte'
    
    test_images, test_labels = load_images()
    
    np.save('test_images', test_images)
    np.save('test_labels', test_labels)
    
    return train_images, train_labels, test_images, test_labels

def load_pickle():
    train_images = np.load('images/training_images.npy')
    train_labels = np.load('images/training_labels.npy')
    
    test_images = np.load('images/test_images.npy')
    test_labels = np.load('images/test_labels.npy')
    
    
    return train_images, train_labels, test_images, test_labels
    
    
if not os.path.isfile('images/training_images.npy'):
    print ('creating pickle.')
    train_images, train_labels, test_images, test_labels = make_pickle()
    
    print ('verifying pickle...')
    
    Vtrain_images, Vtrain_labels, Vtest_images, Vtest_labels = load_pickle()
    
    if train_images.all() == Vtrain_images.all():
        print ('training images verified.')
    else:
        print ('training images pickle corrupt.')
        
    if train_labels.all() == Vtrain_labels.all():
        print ('training labels verified.')
    else:
        print ('training labels pickle corrupt.')
        
    if test_images.all() == Vtest_images.all():
        print ('test images verified.')
    else:
        print ('test images pickle corrupt.')    

    if test_labels.all() == Vtest_labels.all():
        print ('test labels verified.')
    else:
        print ('test labels pickle corrupt.')

    

    
    

