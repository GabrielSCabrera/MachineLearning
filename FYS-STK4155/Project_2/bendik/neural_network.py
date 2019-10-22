# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:20:13 2019

@author: Bendik
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


np.random.seed(1337)
    

def next_layer(w_i, b_i, y_i1, func):
    """
    A function that finds the y-vector for the next layer.
    """
    y_i = func(np.dot(w_i,y_i1) + b_i)
    return y_i

def sigmoid(x):
    """
    The Sigmoid-function.
    """
    f = 1/(1 + np.exp(-x))
    return f

def find_output(inp, ws, bs, func):
    """
    A function that goes through all the layers and nodes, and finds the output.
    """
    ys = [inp] #a list to hold the y-values
    
    for i in range(len(bs)):
        yi = next_layer(ws[i], bs[i], ys[i], func)
        ys.append(yi)
        
    return ys[-1]

def set_up(n_input_nodes = 164, n_output_nodes = 10, n_hidden = 2, n_nodes_hidden = [12, 12]):
    """
    A function that sets up all the vector and matrices we will need with random values between -1 and 1.
    n_hidden is the number of hidden layers
    n_input_nodes is the number of nodes for the input
    n_output_nodes is the number of nodes for the output
    n_nodes_hidden is the number of nodes for each of the hidden layers
    """
    
    inp = (np.random.rand(n_input_nodes) - 0.5)*2 #the input
    
    ws = [] #a list to hold the w-matrices
    n_nodes = [n_input_nodes] + n_nodes_hidden + [n_output_nodes]
    for i in range(n_hidden+1):
        wi = (np.random.rand(n_nodes[i+1],n_nodes[i]) - 0.5)*2
        ws.append(wi)
        
    bs = [] #a list to hold the b-vectors
    for i in range(n_hidden+1):
        bi = (np.random.rand(n_nodes[i+1]) - 0.5)*2
        bs.append(bi)
    
    return inp, ws, bs

def cost_func(desired_output, found_output):
    cost = np.sum((found_output - desired_output)**2)/2
    return cost

def get_handwritten_data():
    
    # display images in notebook
#    %matplotlib inline
    plt.rcParams['figure.figsize'] = (12,12)


    # download MNIST dataset
    digits = datasets.load_digits()
    
    # define inputs and labels
    inputs = digits.images
    labels = digits.target
    
    print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
    print("labels = (n_inputs) = " + str(labels.shape))
    
    
    # flatten the image
    # the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
    n_inputs = len(inputs)
    inputs = inputs.reshape(n_inputs, -1)
    print("X = (n_inputs, n_features) = " + str(inputs.shape))
    
    
    # choose some random images to display
    indices = np.arange(n_inputs)
    random_indices = np.random.choice(indices, size=5)
    
    for i, image in enumerate(digits.images[random_indices]):
        plt.subplot(1, 5, i+1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title("Label: %d" % digits.target[random_indices[i]])
    plt.show()
    
    return inputs, labels

if __name__ == "__main__":
    
    #inputs the handwritten data
    inputs, labels = get_handwritten_data()
    #generates arrays based upon the handwritten data
    inp, ws, bs = set_up(n_input_nodes=len(inputs[0]), n_output_nodes=10)
    
    #splits into training and testing data
    train_size = 0.8
    test_size = 1 - train_size
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, train_size=train_size, test_size=test_size)
    
    out = find_output(inputs_train[0], ws, bs, sigmoid)
    print(out)
    print(out/np.sum(out))
    print(np.sum(out/np.sum(out)))
    
    




















