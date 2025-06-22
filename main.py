from lantern import Linear, MSELoss, SGD, GradScheduler, Sequential, use_cuda
import numpy as np

print("Starting")

l = Sequential(
    Linear(3, 2),
    Linear(2, 1),
)
l.to(np.float32)
gs = GradScheduler(l.parameters(), 2)

optim = SGD(l.parameters(), lr = 0.001, maximize=False, grad_clip=10)
print("Training")
#optim.to(np.float64)
for i in range(4000):
    v = l(np.random.randint(3, 6, (3,)))
    y = np.array([15])
    loss = MSELoss(v, y)
    l.add_grad(loss)
    optim.step()
    print(f"Output: {v[0]:.9f} - Expected: {y} - Loss: {loss} - Epoch {i+1}")

print(f"\n\nInput: 3, 4, 5\nOutput: {l(np.array([3, 4, 5]))}")
