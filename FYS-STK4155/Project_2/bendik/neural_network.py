# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:20:13 2019

@author: Bendik
"""

import numpy as np
    

def next_node(w_i, b_i, y_i1, func):
    """
    A function that finds the y-vector for the next node.
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
    A function that goes through all the nodes, and finds the output.
    """
    ys = [inp] #a list to hold the y-values
    
    for i in range(len(bs)):
        yi = next_node(ws[i], bs[i], ys[i], func)
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

if __name__ == "__main__":
    inp, ws, bs = set_up()
    
    outp = find_output(inp, ws, bs, sigmoid)
    print(inp)
    print(outp)





















