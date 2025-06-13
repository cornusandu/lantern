import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional

array = np.array

class Parameter:
    def __init__(self, dim: Optional[tuple] = None, vals: Optional[np.ndarray] = None, dtype=np.float32):
        if not dim and vals is None:
            raise ValueError("Please specify a shape or parse a list of values.")
        self.dim = dim or vals.shape
        self.vals = vals if vals is not None else np.zeros(shape=dim, dtype=dtype)

    def __getitem__(self, item):
        return self.vals[item]

    def __setitem__(self, key, value):
        self.vals[key] = value

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return self.vals + other
        else:
            return self.vals + other.vals

    def __radd__(self, other):
        # This is called when other + self, if other’s __add__ doesn’t handle Parameter
        if isinstance(other, np.ndarray):
            return other + self.vals
        else:
            if isinstance(other, Parameter):
                return self.vals + other.vals
            else:
                raise NotImplementedError()

    def __array__(self, dtype=None):
        # Called by np.asarray(), ufuncs, etc.
        return self.vals if dtype is None else self.vals.astype(dtype)

    def __iadd__(self, other):
        if isinstance(other, np.ndarray):
            self.vals += other
        else:
            # assume other is Parameter or similar
            self.vals += np.asarray(other)
        return self

class Module:
    def __init__(self):
        self.training = True
        self.modules = []
        self.dtype = np.float32
    def parameters(self):
        p = []
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                p.extend(v.parameters())
            elif isinstance(v, Parameter):
                p.append(v)
        return p
    def forward(self, x):
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(None, np.ones(shape=(in_features, out_features)))
        self.bias = Parameter(None, np.zeros(shape=out_features))

    def forward(self, x):
        outputs = np.zeros(self.out_features, dtype=self.dtype)
        for i in range(self.out_features):
            v = 0
            for o in range(self.in_features):
                v += self.weights[o, i] * x[o]
            outputs[i] = v
        return outputs + self.bias

def MSELoss(output, target):
    loss = np.mean((output - target)**2)
    grad = 2*(output - target) / output.size
    return loss, grad

class Optim:
    def __init__(self): pass

class SGD(Optim):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        super(SGD, self).__init__()
        self.params   = params
        self.lr       = lr
        self.momentum = momentum
        self.vel = {id(p): np.zeros_like(p.vals) for p in params}

    def step(self):
        for p in self.params:
            g = p.grad
            v = self.vel[id(p)]
            v_new = self.momentum * v + g
            p.vals -= self.lr * v_new
            self.vel[id(p)] = v_new