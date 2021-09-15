import numpy as np
from scipy.optimize import optimize

eps = 1e-3  # termination cov_cri
x_0 = [0, 0]  # initial
k = 1  # count
soln = []  # err
itr = []  # itt
error = 1  # conv
a = 0.0001  # alpha


def f(x):
    return (x[0]) ** 2 + 12 * x[0] * x[1] - 8 * x[0] + 10 * x[1] ** 2 + 4 - 14 * x[1]


def grad(x):
    return np.arrey([10 * x[0] + 12 * x[1] - 8, 12 * x[0] + 20 * x[1] - 14])


# def line_search(x):
#     a = 1.
#     d = - g(x)
#     phi = lambda a, x: f(x) - a * 0.8 * (np.transpose(grad(x)) * d
#     while phi(a, x) > f(x - a * np.array(grad(x))):
#         a = 0.5 * a
#     return a

while error >= eps:
    x_k = x_0
    x_0 = x_k - a * grad(x_k)
    error = abs(max(x_0-x_k))
    conv=np.linalg.norm(grad(x_0))
    soln.append(float(error))

    # soln.append(x)
    # error = np.linalg.norm(g(x))
x = np.array([-2*x_0[0]-3*x_0[1]+1,x_0[0],x_0[0]])
print(x)
