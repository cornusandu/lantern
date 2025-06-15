from lantern import Linear, MSELoss, SGD, GradScheduler, Sequential
import numpy as np

l = Sequential(
    Linear(3, 6),
    Linear(6, 1),
    Linear(1, 1)
)
#l.to(np.float16)
gs = GradScheduler(l.parameters(), 2)

optim = SGD(l.parameters(), lr = 0.0001)
#optim.to(np.float64)
for i in range(1000000):
    v = l(np.random.uniform(-10, 10, (3,)))
    y = np.array([15])
    loss = MSELoss(v, y)
    l.add_grad(loss)
    optim.step()
    print(f"Output: {v[0]:.9f} - Expected: {y} - Loss: {loss}")

print(f"\n\nInput: 3, 4, 5\nOutput: {l(np.array([3, 4, 5]))}")
