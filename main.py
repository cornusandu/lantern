from lantern import Linear, MSELoss, SGD, GradScheduler, Sequential
import numpy as np

l = Sequential(
    Linear(3, 2),
    Linear(2, 1),
    Linear(1, 1)
)
gs = GradScheduler(l.parameters(), 2)

optim = SGD(l.parameters(), lr = 0.03)
optim.to(np.float64)
for i in range(100000):
    v = l(np.array([3, 4, 5]))
    y = np.array([15])
    loss = MSELoss(v, y)
    l.add_grad(loss)
    optim.step()
    print(f"Output: {v[0]:.9f} - Expected: {y} - Loss: {loss}")

print(f"\n\nInput: 3, 4, 5\nOutput: {l(np.array([3, 4, 5]))}")
