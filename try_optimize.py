import numpy as np

def calc(x, y, X):
    dataset = np.array([x, y, -x * X, -y * X]).T
    data = np.c_[dataset, np.ones(dataset.shape[0])]
    return np.linalg.lstsq(data, X, rcond=None)[0]


def XY(x, y, b):
    return (
        (b[0] * x + b[1] * y + b[2]) / (b[6] * x + b[7] * y + 1),
        (b[3] * x + b[4] * y + b[5]) / (b[6] * x + b[7] * y + 1)
    )


x = np.array([2, 9, 1, 1, 1, 2, 2, 2, 3, 3])
y = np.array([8, 1, 4, 1, 2, 1, 2, 3, 2, 3])
X = np.array([3, 3, 2, 2, 2, 1, 1, 1, 2, 4])
Y = np.array([1, 8, 1, 6, 1, 2, 3, 2, 3, 2])
b = np.array([2., 4., 3., 6., 5., 7., 4., 9.])
# X, Y = XY(x, y, b)
# print(calc(x, y, Y))
# print(Y)
a = np.array([4., -17.])
f = lambda x, y, a, F: a[0] * x + a[1] * y - F
dataset = np.array([x, y]).T
data = np.c_[dataset, np.ones(dataset.shape[0])]
f_array = f(x, y, a, 0)
a1 = np.array([[1, 2], [2, 3], [3, 4]])
a2 = np.array([[2, 4], [4, 6], [6, 8]])
print(a1 + a2)
