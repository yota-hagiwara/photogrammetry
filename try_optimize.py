import numpy as np
from scipy.optimize import curve_fit


def fn(x, a, b, c, d, e):
    return (a * x[0] + b * x[1] + c) / (d * x[0] + e * x[1] + 1)


x = np.array([
    [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
    [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
])
b = np.array([1.321, 9.22, 4.3, 2.7, 12.1])
popt, pcov = curve_fit(fn, x, np.array(fn(x, b[0], b[1], b[2], b[3], b[4])))
print(b)
print(popt)
