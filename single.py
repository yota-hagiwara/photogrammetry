import numpy as np
from math import sin, cos, tan, atan, sqrt

C = 0.01 # 焦点距離


# eはnp.array([x1s, x2s, x3s])のように用意する
def stat(o, e):
    e = np.vstack([np.ones(e.shape[1]), e])
    return np.linalg.lstsq(e.T, o)[0]


def R(omega, phai, kappa):
    return np.array([
        [
            cos(omega) * cos(kappa),
            -cos(omega) * sin(kappa),
            sin(phai)
        ],
        [
            cos(omega) * sin(kappa) + sin(omega) * sin(phai) * cos(kappa),
            cos(omega) * cos(kappa) - sin(omega) * sin(phai) * sin(kappa),
            -sin(omega) * cos(phai)
        ],
        [
            sin(omega) * sin(kappa) - cos(omega) * sin(phai) * cos(kappa),
            sin(omega) * cos(kappa) + cos(omega) * sin(phai) * sin(kappa),
            cos(omega) * cos(phai)
        ]
    ])


def calc_b(x, y, X, Y, Z):
    return np.array([1, 1, 1, 1, 1, 1, 1, 1])


def calc_external(b, Z_m=0):
    global C
    omega = atan(C * b[7])
    phai = atan(-C * b[6] * cos(omega))
    A = np.array([
        1 + tan(phai) * tan(phai),
        b[0] + b[1] * tan(phai) / sin(omega),
        b[3] + b[4] * tan(phai) / sin(omega),
        tan(phai) / (cos(phai) * tan(omega))
    ])
    kappa = atan(-b[3] / b[0] if phai == 0 else 
        b[1] / b[4] if omega == 0 else
            - ((A[0] * A[2] - A[1] * A[3]) / (A[0] * A[1] + A[2] * A[3]))
    )
    Z_0 = C * cos(omega) * sqrt((A[1] * A[1] + A[2] * A[2]) / (A[0] * A[0] + A[3] * A[3])) + Z_m
    X_0 = b[2] - (tan(omega) * sin(kappa) / cos(phai) - tan(phai) * cos(kappa)) * (Z_m - Z_0)
    Y_0 = b[5] - (tan(omega) * cos(kappa) / cos(phai) + tan(phai) * sin(kappa)) * (Z_m - Z_0)
    return X_0, Y_0, Z_0, omega, phai, kappa


def FG(x, y, X, Y, Z, X_0, Y_0, Z_0, omega, phai, kappa):
    global C
    a = R(omega, phai, kappa)
    denominator = (a[2][0] * (X - X_0) + a[2][1] * (Y - Y_0) + a[2][2] * (Z - Z_0))
    return (
        -C * (a[0][0] * (X - X_0) + a[0][1] * (Y - Y_0) + a[0][2] * (Z - Z_0)) / denominator - x,
        -C * (a[1][0] * (X - X_0) + a[1][1] * (Y - Y_0) + a[1][2] * (Z - Z_0)) / denominator - y,
    )


def collinear_condition(X, Y, Z, X_0, Y_0, Z_0, a):
    global C
    return (
        - C * (a[0][0] * (X - X_0) + a[0][1] * (Y - Y_0) + a[0][2] * (Z - Z_0)) / (a[2][0] * (X - X_0) + a[2][1] * (Y - Y_0) + a[2][2] * (Z - Z_0)),
        - C * (a[1][0] * (X - X_0) + a[1][1] * (Y - Y_0) + a[1][2] * (Z - Z_0)) / (a[2][0] * (X - X_0) + a[2][1] * (Y - Y_0) + a[2][2] * (Z - Z_0))
    )


def calc_coefficient(X, Y, Z, X_0, Y_0, Z_0, omega, phai, kappa):
    global C
    a = R(omega, phai, kappa)
    x, y = collinear_condition(X, Y, Z, X_0, Y_0, Z_0, a)
    f_x = (C * a[0][0] + x * a[2][0]) / (a[2][0] * (X - X_0) + a[2][1] * (Y - Y_0) + a[2][2] * (Z - Z_0))
    f_y = (C * a[0][1] + x * a[2][1]) / (a[2][0] * (X - X_0) + a[2][1] * (Y - Y_0) + a[2][2] * (Z - Z_0))
    g_x = (C * a[1][0] + x * a[2][0]) / (a[2][0] * (X - X_0) + a[2][1] * (Y - Y_0) + a[2][2] * (Z - Z_0))
    g_y = (C * a[1][1] + x * a[2][1]) / (a[2][0] * (X - X_0) + a[2][1] * (Y - Y_0) + a[2][2] * (Z - Z_0))
    return (
        f_x,
        f_y,
        (C * a[0][2] + x * a[2][2]) / (a[2][0] * (X - X_0) + a[2][1] * (Y - Y_0) + a[2][2] * (Z - Z_0)),
        x * y / C,
        - x * x * cos(omega) / C - y * sin(omega) - C * cos(omega),
        f_x * (Y - Y_0) - f_y * (X - X_0),
        g_x,
        g_y,
        (C * a[1][2] + x * a[2][2]) / (a[2][0] * (X - X_0) + a[2][1] * (Y - Y_0) + a[2][2] * (Z - Z_0)),
        C + y * y / C,
        x * sin(omega) - x * y * cos(omega) / C,
        g_x * (Y - Y_0) - g_y * (X - X_0)
    )


def taylor(xs, ys, Xs, Ys, Zs, X_0, Y_0, Z_0, omega, phai, kappa):
    F_Cs = np.array([[0 for j in range(7)] for i in range(len(xs))])
    G_Cs = np.array([[0 for j in range(7)] for i in range(len(xs))])
    for i in range(len(xs)):
        x, y, X, Y, Z = xs[i], ys[i], Xs[i], Ys[i], Zs[i]
        coefficients = calc_coefficient(X, Y, Z, X_0, Y_0, Z_0, omega, phai, kappa)
        for j in range(6):
            F_Cs[j] = coefficients[j]
            G_Cs[j] = coefficients[j + 6]
        F_Cs[i][6], G_Cs[i][6] = FG(x, y, X, Y, Z, X_0, Y_0, Z_0, omega, phai, kappa)
    return (F_Cs, G_Cs)


# x, y, X, Yはそれぞれ基準点の数分の要素を持つ座標配列
def single_photogrammetry(xs, ys, Xs, Ys, Zs):
    Z_m = 0 # 平均標高
    b = calc_b(xs, ys, Xs, Ys, Zs) # b1〜b8を求める
    X_0, Y_0, Z_0, omega, phai, kappa = calc_external(b, Z_m)
    
