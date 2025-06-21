import numpy as np
from typing import Optional
import autograd.numpy as anp
from autograd import grad as autograd_grad

array = np.array

class GradScheduler:
    def __init__(self, params: list["Parameter"], max_history: int = 5):
        self.max_history = max_history

        for p in params:
            orig = p.add_grad
            # capture both orig and this specific p in the default args
            def wrapped_add(loss, orig=orig, p=p):
                orig(loss)
                # trim *each* scalar’s history to the last max_history entries
                for hist in p._grads:
                    if len(hist) > self.max_history:
                        del hist[:-self.max_history]
            p.add_grad = wrapped_add

class Parameter:
    def __init__(
        self,
        dim: Optional[tuple] = None,
        vals: Optional[np.ndarray] = None,
        dtype=np.float32
    ):
        if vals is None and dim is None:
            raise ValueError("Please specify a shape or pass explicit values.")
        # stocăm valorile ca numpy array, dar vom folosi autograd.numpy în grad()
        self.vals = (vals if vals is not None
                     else np.zeros(shape=dim, dtype=dtype)).astype(dtype)
        self.dtype = dtype
        # un istoric (v, loss) pentru fiecare element scalar
        self._grads = [[] for _ in self.vals.flat]

    def to(self, dtype=np.float32):
        self.vals = self.vals.astype(dtype)
        self.dtype = dtype

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
        if isinstance(other, np.ndarray):
            return other + self.vals
        elif isinstance(other, Parameter):
            return self.vals + other.vals
        else:
            raise NotImplementedError()

    def __array__(self, dtype=None):
        return self.vals if dtype is None else self.vals.astype(dtype)

    def __iadd__(self, other):
        if isinstance(other, np.ndarray):
            self.vals += other
        else:
            self.vals += np.asarray(other)
        return self

    def add_grad(self, loss: float):
        """
        Înregistrează (valoare, pierdere) pentru fiecare element scalar,
        în ordinea flatten.
        """
        flat_vals = self.vals.ravel()
        for idx, v in enumerate(flat_vals):
            self._grads[idx].append((float(v), float(loss)))

    def grad(self) -> np.ndarray:
        """
        Pentru fiecare element scalar:
         1) Dacă avem <2 puncte în istoric → 0
         2) Dacă ultimii doi v sunt egali   → 0 (nu putem estima pantă)
         3) Altfel:
            m = (L_n - L_{n-1}) / (v_n - v_{n-1})
        """
        grads = np.zeros_like(self.vals, dtype=self.dtype)

        for idx, history in enumerate(self._grads):
            n = len(history)
            if n < 2:
                grads.flat[idx] = 0.0
                continue

            # extragem ultimele două perechi
            (v_prev, L_prev), (v_curr, L_curr) = history[-2], history[-1]

            # evităm divizie cu zero
            dv = v_curr - v_prev
            if dv == 0.0 or not np.isfinite(dv):
                grads.flat[idx] = 0.0
            else:
                grads.flat[idx] = (L_curr - L_prev) / dv

        return grads

class ModuleList:
    def __init__(self, *modules):
        self.modules: list["Module"] = []
        self.modules.extend(list(modules))

    def parameters(self):
        params = []
        for i in self.modules:
            params.extend(i.parameters())
        return params

    def append(self, module: "Module"):
        self.modules.append(module)

    def extend(self, modules: list["Module"]):
        self.modules.extend(modules)

    def insert(self, module: "Module", index: int):
        self.modules.insert(index, module)

    def __getitem__(self, item):
        return self.modules[item]

    def __setitem__(self, key, value):
        self.modules[key] = value

    def __len__(self):
        return len(self.modules)

class Sequential:
    def __init__(self, *modules):
        self.modules = ModuleList(*modules)

    def parameters(self):
        return self.modules.parameters()

    def forward(self, x):
        for i in self.modules:
            x = i(x)
        return x

    def add_grad(self, loss):
        for i in self.parameters():
            i.add_grad(loss)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, dtype = np.float32):
        for i in self.parameters():
            i.to(dtype)

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
    def add_grad(self, loss):
        for i in self.parameters():
            i.add_grad(loss)

    def to(self, dtype = np.float32):
        for i in self.parameters():
            i.to(dtype)

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
    return loss

class Optim:
    def __init__(self): pass

class SGD(Optim):
    """
    Stochastic Gradient Descent optimizer.

    Args:
        parameters (list[Parameter]): List of model parameters to optimize.
        lr (float): Learning rate.
    """
    def __init__(self, parameters, lr: float = 0.01, maximize: bool = False):
        self.parameters = list(parameters)
        self.lr = lr
        self.maxim = maximize
        self.sign = -1 if maximize else 1
        self.dtype = np.float32

    def to(self, dtype = np.float32):
        self.dtype = dtype

    def step(self):
        """
        Perform a single optimization step:
        - Compute gradients for each parameter using its gradient history.
        - Update parameter values: w = w - lr * grad
        - Clear recorded gradient history for the next iteration.
        """
        for p in self.parameters:
            # Compute current gradient for this parameter
            grad = p.grad()
            flat = grad.ravel()
            for idx in range(flat.size):
                if flat[idx] == 0.0:
                    flat[idx] = np.random.uniform(-0.0000001, 0.0000001)
            grad = flat.reshape(grad.shape)
            p.vals -= self.lr * grad * self.sign

    def zero_grad(self):
        """
        Explicitly clear gradients history without performing an update.
        """
        for p in self.parameters:
            p._grads = [[] for i in p.vals.flat]
