import numpy as np
from typing import Optional

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
    def __init__(self, dim: Optional[tuple] = None, vals: Optional[np.ndarray] = None, dtype=np.float32):
        if not dim and vals is None:
            raise ValueError("Please specify a shape or parse a list of values.")
        self.vals = vals if vals is not None else np.zeros(shape=dim, dtype=dtype)
        self.dtype = dtype
        # one history list per flattened element
        self._grads = [[] for _ in range(self.vals.size)]

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

    def add_grad(self, loss: np.float32):
        # record (value, loss) for each scalar, in flat order
        flat_vals = self.vals.ravel()
        for idx, v in enumerate(flat_vals):
            self._grads[idx].append((float(v), loss))

    def grad(self, dtype=np.float32) -> np.ndarray:
        """
        Estimează gradientul pentru fiecare element din self.vals
        folosind metoda diferenței finite pe ultimele două valori din istoric.

        Istoricul self._grads este o listă de liste, unde fiecare sub-listă conține
        perechi (valoare_parametru, pierdere). Păstrăm doar ultimele 2 pentru a calcula
        panta locală: (l_cur - l_prev) / (w_cur - w_prev).
        """
        grads = np.zeros_like(self.vals, dtype=dtype)
        # self._grads este o listă de sub-liste corespunzătoare fiecărui element din self.vals
        for idx, hist in enumerate(self._grads):
            if len(hist) < 2:
                # nu avem suficiente puncte pentru a estima derivata
                grads.flat[idx] = 0.0
            else:
                # luăm ultimele două puncte
                (w_prev, l_prev), (w_cur, l_cur) = hist[-2], hist[-1]
                denom = w_cur - w_prev
                # dacă diferența în parametru este zero, asumăm gradient nul
                grads.flat[idx] = (l_cur - l_prev) / denom if denom != 0 else 0.0
        return grads

    def _grad_old(self, dtype=None) -> np.ndarray:
        dtype = dtype or self.dtype
        grads_flat = np.zeros(self.vals.size, dtype=dtype)

        for idx, hist in enumerate(self._grads):
            if len(hist) < 2:
                grads_flat[idx] = 0.0
            else:
                try:
                    # Convertim temporar la float32 dacă e float16 (incompatibil cu lstsq)
                    safe_dtype = np.float32 if dtype == np.float16 else dtype
                    w_vals = np.array([w for w, _ in hist], dtype=safe_dtype)
                    l_vals = np.array([l for _, l in hist], dtype=safe_dtype)

                    if np.any(np.isnan(w_vals)) or np.any(np.isinf(w_vals)) or \
                            np.any(np.isnan(l_vals)) or np.any(np.isinf(l_vals)):
                        grads_flat[idx] = 0.0
                        continue

                    A = np.vstack([w_vals, np.ones_like(w_vals)]).T
                    result = np.linalg.lstsq(A, l_vals, rcond=None)[0]
                    a = result[0]

                    grads_flat[idx] = a.astype(dtype)  # convertim înapoi dacă e nevoie
                except np.linalg.LinAlgError:
                    grads_flat[idx] = 0.0

        return grads_flat.reshape(self.vals.shape)

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
            grad = p.grad(self.dtype)
            # Update parameters in-place
            p.vals -= self.lr * grad * self.sign

    def zero_grad(self):
        """
        Explicitly clear gradients history without performing an update.
        """
        for p in self.parameters:
            p._grads = [[] for _ in p.vals]