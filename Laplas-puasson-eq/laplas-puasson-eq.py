#REQUIREMENTS
import subprocess
import sys
try:
    import numpy as np
    from numpy.linalg import solve
    import matplotlib.pyplot as plt
    from math import exp
except:
    try:
        def install(package): subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        install('numpy')
        install('matplotlib')
    except:
        print('ОШИБКА. НЕВОЗМОЖНО УСТАНОВИТЬ НЕОБХОДИМЫЕ ЗАВИСИМОСТИ. Дайте /root права или установите \n - numpy,\n - matplotlib \n сами')
        quit()

eps_0 = 8.85418781762039e-12
def var_input(X1=1, Xn=10, Y1=1, Yn=10, n=10, m =10 ):
    X1 = input("Введите значение x1, например 1: ")
    while not X1: X1 = input("Введите значение x1, например 1: ")
    X1 = float(X1)

    Xn = input("Введите значение xn, например 10: ")
    while not Xn: Xn = input("Введите значение xn, например 10: ")
    Xn = float(Xn)

    Y1 = input("Введите значение y1, например 1: ")
    while not Y1: Y1 = input("Введите значение y1, например 1: ")
    Y1 = float(Y1)

    Yn = input("Введите значение yn, например 10: ")
    while not Yn: Yn = input("Введите значение yn, например 10: ")
    Yn = float(Yn)

    n = input("Введите значение n, например 10: ")
    while not n: n = int(input("Введите значение n, например 10: "))
    n = int(n)

    m = input("Введите значение m, например 10: ")
    while not m: m = input("Введите значение m, например 10: ")
    m = int(m)
    return X1, Xn, Y1, Yn, n, m


def fun_input():
    sigma = str(input("Введите функцию правой части уравнения Пуассона, например exp(-x**2-y**2), 1/\u03B5_0 учтено: "))
    while len(sigma)==0: sigma = str(input("Введите функцию правой части уравнения Пуассона, например exp(-x**2-y**2), 1/\u03B5_0 учтено: "))
    g1 = str(input("Введите функцию g1 на левой границе, например x-y+5: "))
    while len(g1)==0: g1 = str(input("Введите функцию g1 на левой границе, например x-y+5: "))
    g2 = str(input("Введите функцию g2 на правой границе, например x-y+5 : "))
    while len(g2)==0: g2 = str(input("Введите функцию g2 на правой границе, например x-y+5 : "))
    g3 = str(input("Введите функцию g3 на верхней границе, например x-y+5 : "))
    while len(g3)==0: g3 = str(input("Введите функцию g3 на верхней границе, например x-y+5 : "))
    g4 = str(input("Введите функцию g4 на нижней границе, например x-y+5 : "))
    while len(g4)==0:  g4 = str(input("Введите функцию g4 на нижней границе, например x-y+5 : "))
    return sigma, g1, g2, g3, g4


def two_in_one(i, j, n):
    return j * n + i


def borders(A, B, n, m, x, y, g1, g2, g3, g4):
    k3 = 0
    for j in range(0, m):
        for i in range(0, n):
            k3 = two_in_one(i, j, n)
            x, y = X[i], Y[j]
            if i == 0:
                A[k3][k3] = 1
                B[k3] = eval(g1)
            if i == n - 1:
                A[k3][k3] = 1
                B[k3] = eval(g2)
            if j == 0:
                A[k3][k3] = 1
                B[k3] = eval(g3)
            if j == m - 1:
                A[k3][k3] = 1
                B[k3] = eval(g4)
    return A, B


def inside(A, B, n, m, X, Y, dX, dY, sigma):
    k1 = k2 = k3 = k4 = k5 = 0
    for j in range(1, m - 1):
        for i in range(1, n - 1):
            k1 = two_in_one(i - 1, j, n)
            k2 = two_in_one(i + 1, j, n)
            k3 = two_in_one(i, j, n)
            k4 = two_in_one(i, j - 1, n)
            k5 = two_in_one(i, j + 1, n)
            A[k3][k1] = 1 / (dX ** 2)
            A[k3][k2] = 1 / (dX ** 2)
            A[k3][k3] = -2 / (dX ** 2) - 2 / (dY ** 2)
            A[k3][k4] = 1 / (dY ** 2)
            A[k3][k5] = 1 / (dY ** 2)
            x, y = X[i], Y[j]
            B[k3] = eval(sigma) / eps_0
    return A, B


X1, Xn, Y1, Ym, n, m = var_input()
sigma, g1, g2, g3, g4 = fun_input()

A = [[0 for j in range(0, n * m)] for i in range(0, n * m)]
B = [0 for i in range(0, m * n)]

dX = (Xn - X1) / (n - 1)
dY = (Ym - Y1) / (m - 1)

X = [i * dX + X1 for i in range(0, n)]
Y = [j * dY + Y1 for j in range(0, m)]

A, B = borders(A, B, n, m, X, Y, g1, g2, g3, g4)
A, B = inside(A, B, n, m, X, Y, dX, dY, sigma)
U = solve(A, B)

U_ij = []
k = 0
for i in range(n):
    U_ij.append([])
    for j in range(m):
        U_ij[i].append(U[k])
        k += 1


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")

for i in range(len(Y)):
    X_ = np.array(X)
    Y_ = np.array([Y[i] for j in range(len(Y))])
    Z_ = np.array(U[0 + i * len(Y):len(Y) + len(Y) * i])
    ax.plot3D(X_, Y_, Z_, 'green')


print(X)
print(Y)
print(U)
plt.show()