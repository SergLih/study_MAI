from prettytable import PrettyTable
import numpy as np


def tma(a, b, c, d):
    shape = len(d)
    p = [-c[0] / b[0]]
    q = [d[0] / b[0]]
    x = [0] * (shape + 1)
    for i in range(1, shape):
        p.append(-c[i] / (b[i] + a[i] * p[i - 1]))
        q.append((d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1]))
    for i in reversed(range(shape)):
        x[i] = p[i] * x[i + 1] + q[i]
    return x[:-1]

def finite_difference_method(a, b, h):
    cur_pos = a
    x = []
    y = []
    while round(cur_pos, 6) <= b:
        y.append(3*cur_pos + np.exp(-2*cur_pos))
        x.append(cur_pos)
        cur_pos += h
    n = len(x)
    first_conditions = [0.5, 1, 4.5]
    second_conditions = [1, 0, 3 - 2 * np.exp(-4)]
    if first_conditions[0] != 0:
        a = [None] * n
        b = [None] * n
        c = [None] * n
        d = [0] * n
        a[0] = 0
        b[0] = -first_conditions[0]/h+first_conditions[1]
        c[0] = first_conditions[0]/h
        d[0] = first_conditions[2]
        s = 1
        for i in range(s, n - 1):
            a[i] = 1*g(x[i]) - p(x[i]) * h / 2
            b[i] = -2*g(x[i]) + (h ** 2) * q(x[i])
            c[i] = 1*g(x[i]) + p(x[i]) * h / 2
        if second_conditions[0] != 0:
            a[n-1] = -second_conditions[0] / h
            b[n-1] = second_conditions[0] / h + second_conditions[1]
            c[n-1] = 0
            d[n-1] = second_conditions[2]
        else:
            a[n-1] = 1*g(x[n-1])-p(x[n-1]*h/2)
            b[n-1] = -2*g(x[n-1]) + h**2*q(x[n-1])
            c[n-1] = 0
            d[n-1] = -(1*g(x[n-1]) + p(x[n-1])*h/2)*second_conditions[2]/second_conditions[1]
    else:
        a = [None] * (n-1)
        b = [None] * (n-1)
        c = [None] * (n-1)
        d = [0] * (n-1)
        a[0] = 0
        b[0] = -2+h**2*q(x[1])
        c[0] = 1 + p(x[1])*h/2
        d[0] = (p(x[1])*h/2-1)*first_conditions[2]/first_conditions[1]
        for i in range(1, n - 1):
            a[i] = 1*g(x[i]) - p(x[i]) * h / 2
            b[i] = -2*g(x[i]) + (h ** 2) * q(x[i])
            c[i] = 1*g(x[i]) + p(x[i]) * h / 2

        if second_conditions[0] != 0:
            a[n-2] = -second_conditions[0] / h
            b[n-2] = second_conditions[0] / h + second_conditions[1]
            c[n-2] = 0
            d[n-2] = second_conditions[2]
        else:
            a[n-2] = 1*g(x[n-1])-p(x[n-1]*h/2)
            b[n-2] = -2*g(x[n-1]) + h**2*q(x[n-1])
            c[n-2] = 0
            d[n-2] = -(1*g(x[n-1]) + p(x[n-1])*h/2)*second_conditions[2]/second_conditions[1]

    roots = tma(a, b, c, d)

    return roots, x


def runge_kutt_method(x, step, first_y, first_z):
    y = [first_y]
    z = [first_z]
    delta_y = []
    delta_z = []
    for i in range(len(x)-1):
        K = []
        L = []
        K.append(step*z[i])
        L.append(step*f(x[i], y[i], z[i]))
        for j in range(1, 3):
            K.append(step*(z[i] + 0.5*L[j-1]))
            L.append(step*f(x[i] + 0.5*step, y[i] + 0.5*K[j-1], z[i] + 0.5*L[j-1]))
        K.append(step * (z[i] + L[2]))
        L.append(step * f(x[i] + step, y[i] + K[2], z[i] + L[2]))
        delta_y.append(1/6*(K[0] + 2*K[1] + 2*K[2] + K[3]))
        delta_z.append(1/6*(L[0] + 2*L[1] + 2*L[2] + L[3]))
        y.append(y[len(y)-1] + delta_y[len(delta_y)-1])
        z.append(z[len(z)-1] + delta_z[len(delta_z)-1])
    delta_y.append(None)
    delta_z.append(None)
    return y, z, delta_y, delta_z


def shooting_method(a, b, h):
    x = []
    y = []
    x = list(np.arange(a, b + h, h))
    first_conditions = [0.5, 1, 4.5]
    second_conditions = [1, 0, 3 - 2 * np.exp(-4)]
    right_derivative = second_conditions[2]/second_conditions[0]
    fst_derivatives = [1, 2]
    eps = 0.01

    penultimate_first_derivative = fst_derivatives[len(fst_derivatives) - 2]
    last_first_derivative = fst_derivatives[len(fst_derivatives) - 1]

    first_y = (first_conditions[2] - first_conditions[0] * penultimate_first_derivative) / first_conditions[1]
    y1, z1, delta_y1, delta_z1 = runge_kutt_method(x, h, first_y, penultimate_first_derivative)

    first_y = (first_conditions[2] - first_conditions[0] * last_first_derivative) / first_conditions[1]
    y2, z2, delta_y2, delta_z2 = runge_kutt_method(x, h, first_y, last_first_derivative)

    y = np.copy(y2)
    fst_derivatives.append(last_first_derivative - (last_first_derivative - penultimate_first_derivative)*(z2[len(z2) - 1] - right_derivative)/
                           (z2[len(z2) - 1] - z1[len(z1) - 1]))
    last_derivative = z2[len(z2)-1]

    while abs(last_derivative - right_derivative) > eps:
        first_y = (first_conditions[2]-first_conditions[0]*fst_derivatives[len(fst_derivatives)-1])/first_conditions[1]
        first_derivative = fst_derivatives[len(fst_derivatives)-1]

        cur_y, cur_z, delta_y, delta_z = runge_kutt_method(x, h, first_y, first_derivative)

        penultimate_first_derivative = fst_derivatives[len(fst_derivatives)-2]
        last_first_derivative = fst_derivatives[len(fst_derivatives)-1]
        y = np.copy(cur_y)
        fst_derivatives.append(last_first_derivative - (last_first_derivative - penultimate_first_derivative)/
                               (cur_z[len(cur_z)-1] - last_derivative)*(cur_z[len(cur_z)-1]-right_derivative))

        last_derivative = cur_z[len(cur_z)-1]
    return y, x


if __name__=="__main__":
    
    f         = lambda x, y, y_der: (4 * y - 4 * x * y_der) / (2 * x + 1)
    precise_y = lambda x: 3 * x + np.exp(-2 * x)
    p         = lambda x: 4 * x
    q         = lambda x: -4
    g         = lambda x: 2 * x + 1

    a = 1
    b = 2
    h  = 0.01 

    y, x       = shooting_method(a, b, h)
    y1_1, x1_1 = shooting_method(a, b, h/2)
    table = PrettyTable()

    y1,   x1     = finite_difference_method(a, b, h)
    y2_2, x2_2   = finite_difference_method(a, b, h/2)
    print("-"*39, "Shooting method", "-"*39)
    table = PrettyTable()
    table._set_field_names(["k", "x", "y", "precise_y", "eps", "runge eps"])
    right_y = list(map(precise_y, x1))
    p = 2
    for i in range(len(x)-1):
        table.add_row([i, x[i], y[i], right_y[i], abs(y[i]-right_y[i]), abs(y[i] - y1_1[i]) / (2**p - 1)])
    print(table)

    table = PrettyTable()
    print("-"*34, "Finite difference method", "-"*34)
    table._set_field_names(["k", "x", "y", "precise_y", "eps", "runge eps"])
    p = 2
    for i in range(len(x1)):
        table.add_row([i, x1[i], y1[i], right_y[i], abs(y1[i]-right_y[i]), abs(y1[i] - y2_2[i]) / (2**p - 1)])
    print(table)
