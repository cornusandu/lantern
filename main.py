import numpy as np
import lantern

print("Starting")
l = lantern.ReLU()
print(np.arange(-5, 5))
print(l(np.arange(-5, 5)))
