# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# Let N be the number of inputs, L the number of hidden neurons and M the
# number of outputs. Suppose you have Q different inputs with known outputs
# for the learning stage. w^h is a L x N matrix and w^o is a M x L matrix
# filled originally with random real numbers between -1 and 1.
# x is a Q x N matrix with the values of the inputs of the known patterns.
# d is a Q x M matrix with the values of the outputs of the known patterns.
# f is a sigmoid function (in our case f(x) = 1/(1+e^(-alpha*x)))

def f(x, alpha) :
    return 1 / (1 + np.exp(-alpha*x))

def n_significant_figures(num, n) :
    return float(format(num, '.'+str(n)+'g'))

def train_NN(x, d, alpha) :
    Q = np.size(x, axis=0)
    N = np.size(x, axis=1)
    M = np.size(d, axis=1)
    L = N + M
    w_h = np.random.uniform(-1, 1, (L, N))
    w_o = np.random.uniform(-1, 1, (M, L))
    
    error = np.Inf
    previous_error = 0
    vf = np.vectorize(f)
    
    while(error != previous_error) :
        for j in np.arange(Q) :
            net_h = w_h @ x[j].T
            y_h = vf(net_h, alpha)            
            net_o = w_o @ y_h
            y = vf(net_o, alpha)
    
            delta_o = np.multiply((d[j].T - y), np.multiply(y, (1-y)))
            delta_h = np.multiply((delta_o.T @ w_o).T, np.multiply(y_h, 1-y_h))
            w_h = w_h + alpha*delta_h @ x[j]
            w_o = w_o + alpha*delta_o @ (y_h.T) 
        
        previous_error = error
        error = n_significant_figures(np.linalg.norm(delta_o), 5)
    
    return w_h, w_o, error
#%%
def predict(inputs, w_h, w_o, alpha) :
    vf = np.vectorize(f)
    
    net_h = w_h @ inputs.T
    y_h = vf(net_h, alpha)            
    net_o = w_o @ y_h
    y = vf(net_o, alpha)
    
    return y
#%%
alpha = 3
data = pd.read_excel("tabla_para_probar.xlsx")
data = data[ data["d1"] != "?" ]
x = np.matrix(data.iloc[:, 0:4])
d = np.matrix(data.iloc[:, 4:6])

w_h, w_o, error = train_NN(x, d, alpha)

#%%
data = pd.read_excel("tabla_para_probar.xlsx")
x = np.matrix(data.iloc[:, 0:4])
d = np.matrix(data.iloc[:, 4:6])
vround = np.vectorize(round)
for i in range(11) :
    print(vround(predict(x[i], w_h, w_o, alpha)).T)
    print(d[i])
    print("\n")