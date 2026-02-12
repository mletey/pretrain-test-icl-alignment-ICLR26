"""
In-context regression tasks data sampler

Mary Letey, January 2026
With thanks to William Tong for original codebase
"""

import numpy as np
import random

# d: token and task dimension
# l: context length 
# n: batch size i.e. number of sequences
# k: number of unique tasks in batch 
    # if k = -1, a unique task will be sampled per sequence (this is the k = infinity case)
    # else, k tasks will be preselected and shared amongst the n sequences
# rho: label noise is N(0, rho)
# Ctask: dxd task covariance matrix
# function_type: allows for single index functions
    # default value = 1, y = x dot w. THIS IS ALMOST ALWAYS USED
    # any non-zero value a, y = (x dot w)^a
    # value 0, y = tanh(x dot w)

class task_sampler:
    def __init__(self, d, l, n, k, rho, Ctask, function_type=1) -> None:
        self.d = d # DIMENSION
        self.l = l # CONTEXT LENGTH
        self.n = n # NUMBER OF CONTEXTS
        self.k = k # TASK DIVERSITY
        self.rho = rho # LABEL NOISE
        self.Ctask = Ctask # TASK COVARIANCE
        self.rng = np.random.default_rng(None) # SAMPLER OBJECT 

        # Now we fix a set of TASKS which will be sampled from during all other calls to iter or next once this object is instantiated. 
        if self.k != -1:
            self.E = np.random.multivariate_normal(mean=np.zeros(self.d), cov=self.Ctask, size=self.k).T  # Shape: (D, K)
        self.function_type = function_type
        
    def __next__(self):
        if self.k == -1:
            ws = self.rng.multivariate_normal(mean=np.zeros(self.d), cov=self.Ctask, size=self.n)
            ws = ws[:,:,np.newaxis] # batch_size x n_dims x 1 
        else: 
            uniform_ps = np.array([random.randrange(self.k) for _ in range(self.n)])
            ws = np.array([self.E[:,uniform_ps[i]] for i in range(len(uniform_ps))]) 
            ws = ws[:,:,np.newaxis] # batch_size x n_dims x 1 
        
        xs = self.rng.normal(loc=0, scale = 1/np.sqrt(self.d), size=(self.n, self.l + 1, self.d))

        if self.function_type==0:
            ys = np.tanh(xs @ ws) + self.rng.normal(loc=0, scale = np.sqrt(self.rho), size=(self.n, self.l+1, 1))
        else:
            ys = (xs @ ws)**self.function_type + self.rng.normal(loc=0, scale = np.sqrt(self.rho), size=(self.n, self.l+1, 1))

        Z = np.zeros((self.n, self.l + 1, self.d + 1))
        Z[:,:,0:self.d] = xs
        Z[:,:,-1] = ys.squeeze()
        Z[:,-1, self.d] = 0 # padding for final context
        # returns the Z [x,y,x,y]... configuration and the true N+1 value for testing 
        return Z, ys[:,-1].squeeze(), ws

    def __iter__(self):
        return self
    
class task_sampler_fixed_ws:
    def __init__(self, d, l, n, ws, rho, function_type=1) -> None:
        self.d = d # DIMENSION
        self.l = l # CONTEXT LENGTH
        self.n = n # NUMBER OF CONTEXTS
        self.rho = rho # LABEL NOISE
        self.rng = np.random.default_rng(None) # SAMPLER OBJECT 
        self.ws = ws
        self.function_type = function_type
        
    def __next__(self):

        xs = self.rng.normal(loc=0, scale = 1/np.sqrt(self.d), size=(self.n, self.l + 1, self.d))

        if self.function_type==0:
            ys = np.tanh(xs @ self.ws) + self.rng.normal(loc=0, scale = np.sqrt(self.rho), size=(self.n, self.l+1, 1))
        else:
            ys = (xs @ self.ws)**self.function_type + self.rng.normal(loc=0, scale = np.sqrt(self.rho), size=(self.n, self.l+1, 1))

        Z = np.zeros((self.n, self.l + 1, self.d + 1))
        Z[:,:,0:self.d] = xs
        Z[:,:,-1] = ys.squeeze()
        Z[:,-1, self.d] = 0 # padding for final context
        # returns the Z [x,y,x,y]... configuration and the true N+1 value for testing 
        return Z, ys[:,-1].squeeze()

    def __iter__(self):
        return self