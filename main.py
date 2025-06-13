from lantern import Linear, MSELoss, SGD
import numpy as np

l = Linear(3, 1)

optim = SGD(l.parameters())
for i in range(100):
    v = l(np.array([3, 4, 5]))
    y = np.array([15])
    loss = MSELoss(v, y)
    optim.step(loss)
    print(f"Output: {v} - Expected: {y} - Loss: {loss}")

print(f"\n\nInput: 3, 4, 5\nOutput: {l(np.array([3, 4, 5]))}")