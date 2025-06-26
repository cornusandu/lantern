import numpy as np
from typing import Optional
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def use_cuda():
    globals()['np'] = __import__('cupy')



def spline_gradient_r2(x: np.ndarray, y: np.ndarray):
    # Sortare și validare direct pe GPU
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    spline = CubicSpline(x_s, y_s)
    # Derivata la media lui x
    x_mean = np.mean(x_s)
    slope = spline.derivative()(x_mean)
    # R² pe GPU
    y_pred = spline(x_s)
    ss_res = np.sum((y_s - y_pred) ** 2)
    ss_tot = np.sum((y_s - x_s.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return float(slope), float(r_squared)

def linear_regression_slope_r2(x: np.ndarray, y: np.ndarray):
    # Verificări dimensionale
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x și y trebuie să fie vectori unidimensionali de aceeași lungime.")

    # Centrează datele
    x_mean = x.mean()
    y_mean = y.mean()
    x_cent = x - x_mean
    y_cent = y - y_mean

    # Caz degenerat: varianța lui x = 0 → pantă indefinită
    var_x = (x_cent ** 2).sum()
    if var_x == 0:
        raise ValueError("Toate valorile lui x sunt identice; nu se poate calcula panta.")

    # Calculează pantă și intercept
    slope = (x_cent * y_cent).sum() / var_x
    intercept = y_mean - slope * x_mean

    # Predicții și calcul R²
    y_pred = slope * x + intercept
    ss_tot = (y_cent ** 2).sum()
    ss_res = ((y - y_pred) ** 2).sum()

    # Dacă variația lui y = 0, R² este 1 (modelul explică perfect lipsa de variație)
    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = 1.0 - ss_res / ss_tot

    # Returnăm doar ca float Python (evităm return ndarray de dim 0)
    return float(slope), float(r_squared)

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

    __array_priority__ = 1000

    def __init__(
        self,
        dim: Optional[tuple] = None,
        vals: Optional[np.ndarray] = None,
        dtype=np.float32
    ):
        if vals is None and dim is None:
            raise ValueError("Please specify a shape or pass explicit values.")
        # stocăm valorile ca numpy array, dar vom folosi autograd.numpy în grad()
        self.vals: np.ndarray = (vals if vals is not None
                     else np.zeros(shape=dim, dtype=dtype)).astype(dtype)
        self.dtype = dtype
        self.dim = self.vals.shape
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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # înlocuim fiecare Parameter cu ndarray‐ul său intern
        args = [x.vals if isinstance(x, Parameter) else x for x in inputs]
        # apelăm ufunc‐ul original pe ndarray‐uri
        result = getattr(ufunc, method)(*args, **kwargs)
        # dacă ufunc‐ul întoarce un tuple, îl procesăm recursiv
        if isinstance(result, tuple):
            return tuple(type(self)(vals=r) if isinstance(r, type(args[0])) else r for r in result)
        # altfel, întoarcem rezultatul ca ndarray
        return result

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
        grads = np.zeros_like(self.vals.ravel())
        for f, h in enumerate(self._grads):
            x = np.zeros((len(h),))
            y = np.zeros((len(h),))
            for i, (v, l) in enumerate(h): x[i] = v; y[i] = l;
            try:
                if len(x) <= 1:
                    slope, r2 = 0, 0
                elif len(x) > 2:
                    slope, r2 = spline_gradient_r2(x, y)
                else:
                    slope, r2 = linear_regression_slope_r2(x, y)
            except ValueError as _:
                slope = 0
                r2 = 0
            grads[f] = slope
        grads = grads.reshape(self.vals.shape)
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
        return x.dot(self.weights.vals) + self.bias.vals

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return np.where(x > 0, x, 0)

def MSELoss(output, target):
    loss = np.mean(np.power(output - target, 2))
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
    def __init__(self, parameters, lr: float = 0.01, momentum: float | None = None, use_momentum: bool = False, maximize: bool = False, grad_clip = 10.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.maxim = maximize
        self.sign = -1 if maximize else 1
        self.dtype = np.float32
        self.grad_clip = grad_clip
        self.m = []
        self.momentum = momentum or 0.99
        if use_momentum:
            for p in parameters:
                ms = np.zeros_like(p.vals)
                self.m.append(ms)
        self.use_momentum = use_momentum

    def to(self, dtype = np.float32):
        self.dtype = dtype

    def step(self):
        """
        Perform a single optimization step:
        - Compute gradients for each parameter using its gradient history.
        - Update parameter values: w = w - lr * grad
        - Clear recorded gradient history for the next iteration.
        """
        for pi, p in enumerate(self.parameters):
            # Compute current gradient for this parameter
            grad = p.grad()
            flat = grad.ravel()
            for idx in range(flat.size):
                if flat[idx] == 0.0:
                    flat[idx] = np.random.uniform(-0.0000001, 0.0000001)
            grad = flat.reshape(grad.shape)
            grad = np.clip(grad, -self.grad_clip, self.grad_clip)
            if not self.use_momentum:
                p.vals -= grad * self.lr * self.sign
            else:
                self.m[pi] = self.momentum * self.m[pi] - self.lr * grad
                p.vals += self.m[pi]

    def zero_grad(self):
        """
        Explicitly clear gradients history without performing an update.
        """
        for p in self.parameters:
            p._grads = [[] for i in p.vals.flat]
