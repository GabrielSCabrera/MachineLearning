# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:20:13 2019

@author: Bendik
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import time


np.random.seed(1337)
    

def shapes(*args):
    for i in args:
        print(i.shape, end = "")
    print()

def next_layer(w_i, b_i, y_i1, func):
    """
    A function that finds the y-vector for the next layer.
    """
    z_i = (w_i @ y_i1.T).T + b_i
    y_i = func(z_i)
    return y_i

def sigmoid(x):
    """
    The Sigmoid-function.
    """
    f = 1/(1 + np.exp(-x))
    return f

def feed_forward(inp, ws, bs, func):
    """
    A function that goes through all the layers and nodes, and finds the output.
    """
    ys = [inp] #a list to hold the y-values
    
    for i in range(len(bs)):
        yi = next_layer(ws[i], bs[i], ys[i], func)
        ys.append(yi)
    
    prob = ys[-1].T/np.sum(ys[-1], axis=1)
        
    return ys, prob.T

def set_up(n_input_nodes = 164, n_output_nodes = 10, n_hidden = 2, n_nodes_hidden = 12):
    """
    A function that sets up all the vector and matrices we will need with random values between -1 and 1.
    n_hidden is the number of hidden layers
    n_input_nodes is the number of nodes for the input
    n_output_nodes is the number of nodes for the output
    n_nodes_hidden is the number of nodes for each of the hidden layers
    """
    
    inp = (np.zeros(n_input_nodes) + 0.1) #the input
    
    ws = [] #a list to hold the w-matrices
    n_nodes = [n_input_nodes]
    for i in range(n_hidden):
        n_nodes.append(n_nodes_hidden)
    n_nodes.append(n_output_nodes)

    
    for i in range(n_hidden+1):
        wi = (np.random.rand(n_nodes[i+1],n_nodes[i]) - 0.5)*2
        ws.append(wi)
        
    bs = [] #a list to hold the b-vectors
    for i in range(n_hidden+1):
        bi = (np.random.rand(n_nodes[i+1]) - 0.5)*2
        bs.append(bi)
    
    return inp, ws, bs

def loss_func(desired_output, found_output):
    """
    The loss/cost function
    """
    loss = np.sum((found_output - desired_output)**2)/2
    return loss

def get_handwritten_data():
    """
    A function that collects and returns the handwritten data.
    """
    
    # display images in notebook
#    %matplotlib inline
    plt.rcParams['figure.figsize'] = (12,12)


    # download MNIST dataset
    digits = datasets.load_digits()
    
    # define inputs and labels
    inputs = digits.images
    labels = digits.target
    
#    print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
#    print("labels = (n_inputs) = " + str(labels.shape))
    
    
    # flatten the image
    # the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
    n_inputs = len(inputs)
    inputs = inputs.reshape(n_inputs, -1)
#    print("X = (n_inputs, n_features) = " + str(inputs.shape))
    
    
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


def to_categorical_numpy(integer_vector):
    """
    Outputs a given vector in the 1-hot form.
    """
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

def backpropagation(inputs, labels, ws, bs, func):
    """
    A function htat does backpropagation.
    """    
    
    #runs the feed forward thing
    a_h, prob = feed_forward(inputs, ws, bs, func)
    
    errs = [] #the errors
    #error in the final output layer
    err_out = prob - labels
    errs.append(err_out)
    
    #error in the rest of the layers
    for i in range(len(ws)-1):
        err_i = (errs[0] @ ws[-1-i]) * a_h[-2-i] * (1 - a_h[-2-i])
        errs = [err_i] + errs
    
    w_grad = [] #the gradients for the weights
    for i in range(len(ws)):
        w_g_i = a_h[i].T @ errs[i]
        w_grad.append(w_g_i)
    
    b_grad = [] #the gradient for the biases
    for i in range(len(bs)):
        b_g_i = np.sum(errs[i])
        b_grad.append(b_g_i)
    
    return w_grad, b_grad

def predict(X, ws, bs, func):
    probabilities = feed_forward(X, ws, bs, func)[1]
#    print("np.sum(probabilities[:,0])", np.sum(probabilities[:,0]))
#    print("probabilities[:,0]", probabilities[:,0])
#    print("np.sum(probabilities[:,1])", np.sum(probabilities[:,1]))
#    shapes(probabilities)
    pro = np.argmax(probabilities, axis=1)
#    print("pro", pro)
#    print("np.shape(pro)", np.shape(pro))
#    print("")
    return pro


if __name__ == "__main__":
    
    #inputs the handwritten data
    inputs, labels = get_handwritten_data()
    #generates arrays based upon the handwritten data
    inp, ws, bs = set_up(n_input_nodes=len(inputs[0]), n_output_nodes=10, n_hidden=10, n_nodes_hidden=50)
    
    #splits into training and testing data
    train_size = 0.8
    test_size = 1 - train_size
    inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, train_size=train_size, test_size=test_size)
    
    labels_train_onehot, labels_test_onehot = to_categorical_numpy(labels_train), to_categorical_numpy(labels_test)

    print("Old accuracy on training data: " + str(accuracy_score(labels_train, predict(inputs_train, ws, bs, sigmoid))))
    print("Old accuracy on testing data: " + str(accuracy_score(labels_test, predict(inputs_test, ws, bs, sigmoid))))
    print(predict(inputs_test, ws, bs, sigmoid))
    print(min(predict(inputs_test, ws, bs, sigmoid)))
    print(max(predict(inputs_test, ws, bs, sigmoid)))
    print("")
    

    eta = 0.001
    lambd = 0.01
    batch_size = 100
    epochs = 100
    iterations = len(inputs_train) // batch_size
    
    perc = 0
    tot_iter = iterations*epochs
    times = np.zeros(tot_iter)
    counter = 0
    t0 = time()
    dt = 0
    print(f"\t{0:>3d}%", end = "")
    
    data_indices = np.arange(len(inputs_train))    
    for j in range(epochs):
        for i in range(iterations):
            
            chosen_datapoints = np.random.choice(data_indices, size=batch_size, replace=False)
            
            counter += 1
            new_perc = int(100*counter/tot_iter)
            times[counter-1] = time()
            if int(time() - t0) > dt:
                perc = new_perc
                t_avg = np.mean(np.diff(times[:counter]))
                ETA = t_avg*(tot_iter - counter)
                hh = ETA//3600
                mm = (ETA//60)%60
                ss = ETA%60
                msg = f"\r\t{perc:>3d}% – ETA {hh:02.0f}:{mm:02.0f}:{ss:02.0f}"
                print(msg, end = "")
            dt = time() - t0
            
            #runs the backpropagation
            w_grad, b_grad = backpropagation(inputs_train[data_indices], labels_train_onehot[data_indices], ws, bs, sigmoid)
            
            #regularization term gradients
            for i in range(len(w_grad)):
                w_grad[i] += lambd * ws[i].T
            
            #update weight and biases
            for i in range(len(w_grad)):
                ws[i] -= eta * w_grad[i].T
                bs[i] -= eta * b_grad[i]
            
    dt = time() - t0
    hh = dt//3600
    mm = (dt//60)%60
    ss = dt%60
    print(f"\r\t100% – Total Time Elapsed {hh:02.0f}:{mm:02.0f}:{ss:02.0f}")

    print("New accuracy on training data: " + str(accuracy_score(labels_train, predict(inputs_train, ws, bs, sigmoid))))
    print("New accuracy on testing data: " + str(accuracy_score(labels_test, predict(inputs_test, ws, bs, sigmoid))))
    print(predict(inputs_test, ws, bs, sigmoid))
    print(min(predict(inputs_test, ws, bs, sigmoid)))
    print(max(predict(inputs_test, ws, bs, sigmoid)))
    






