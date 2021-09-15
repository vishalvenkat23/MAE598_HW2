import numpy as np
from scipy.optimize import optimize

def f(x):
    return (x[0]) ** 2 + 12 * x[0] * x[1] - 8 * x[0] + 10 * x[1] ** 2 + 4 - 14 * x[1]
def grad(x):
    return np.arrey([10 * x[0] + 12 * x[1] - 8, 12 * x[0] + 20 * x[1] - 14])

eps = 1e-3  # termination
x_0 = [0, 0]
k = 0
soln = [x_0]
x = soln[k]
error = np.linalg.norm(g(x))
# a_s = 0.01
def line_search(x):
    a = 1.
    d = - g(x)
    phi = lambda a, x: f(x) - a * 0.8 * (np.transpose(grad(x)) * d
    while phi(a, x) > f(x - a * np.array(grad(x))):
        a = 0.5 * a
    return a

while error >= eps:
    x = x - a * g(x)
    soln.append(x)
    error = np.linalg.norm(g(x))
