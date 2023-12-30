import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

def eiler_method(a, b, h, first_y, first_y_der):
    i = 0
    x = [a]
    y = [first_y]
    y_der = [first_y_der]
    cur_pos = a + h
    while round(cur_pos, 6) <= b:
        x.append(cur_pos)
        y.append(y[i] + h * y_der[i])
        y_der.append(y_der[i] + h * f(cur_pos, y[i], y_der[i]))
        cur_pos += h
        i += 1
    return y, y_der

def adams_method(a, b, first_y, first_y_der, h):

    x = list(np.arange(a, b + h, h))
    i = len(x)
    y, y_der, delta_y, delta_y_der = runge_kutt_method(x, h, first_y, first_y_der)

    x = list(np.arange(a, b + h, h))
    while i != len(x):
        y.append(y[i-1] + h/24*(55*y_der[i-1] - 59*y_der[i-2] + 37*y_der[i-3] - 9*y_der[i-4]))
        y_der.append(y_der[i-1] + h/24*(55*f(x[i-1], y[i-1], y_der[i-1]) - 59*f(x[i-2], y[i-2], y_der[i-2]) +
                                   37*f(x[i-3], y[i-3], y_der[i-3]) - 9*f(x[i-4], y[i-4], y_der[i-4])))
        i += 1
    return y


def runge_kutt_method(x, h, first_y, first_y_der):
    y = [first_y]
    y_der = [first_y_der]
    delta_y = []
    delta_y_der = []
    for i in range(len(x)-1):
        K = []
        L = []
        K.append(h*y_der[i])
        L.append(h*f(x[i], y[i], y_der[i]))
        for j in range(1, 3):
            K.append(h*(y_der[i] + 0.5*L[j-1]))
            L.append(h*f(x[i] + 0.5*h, y[i] + 0.5*K[j-1], y_der[i] + 0.5*L[j-1]))
        K.append(h * (y_der[i] + L[2]))
        L.append(h * f(x[i] + h, y[i] + K[2], y_der[i] + L[2]))
        delta_y.append(1/6*(K[0] + 2*K[1] + 2*K[2] + K[3]))
        delta_y_der.append(1/6*(L[0] + 2*L[1] + 2*L[2] + L[3]))
        y.append(y[len(y)-1] + delta_y[len(delta_y)-1])
        y_der.append(y_der[len(y_der)-1] + delta_y_der[len(delta_y_der)-1])
    delta_y.append(None)
    delta_y_der.append(None)
    return y, y_der, delta_y, delta_y_der


if __name__ == "__main__":

    f_y = lambda x: (x**2 + 1 / x)*np.exp(x**0.5)
    f = lambda x, y, y_der: y_der / x**0.5 - y / (4 * x**2) * (x + x**0.5 - 8)
    
    y_first = 2*np.exp(1)
    y_der_first = 2*np.exp(1)
    h = 0.1
    a = 1.0
    b = 2.0
    x = list(np.arange(a, b+h, h))
    
    print(" " * 25, "Eiler method")
    p = 2
    y1, y_der1 = eiler_method(a, b, h, y_first, y_der_first)
    y2, y_der2 = eiler_method(a, b, 0.5 * h, y_first, y_der_first)
    line1, = plt.plot(x, y1, label="Eiler method")
    table = PrettyTable()
    table._set_field_names(["k", "x", "y", "analitic y", "eps", "runge romberg precision"])
    for i in range(len(x)):
        table.add_row([i, x[i].round(6), y1[i].round(6), f_y(x[i]).round(6), round(abs(y1[i] - f_y(x[i])), 6), round(abs(y2[2*i]-y1[i])/(2**p-1), 6)])
    print(table)
    print()
    
    x1 = list(np.arange(a, b + 0.5 * h, 0.5 * h))
    y1, y_der1, delta_y1, delta_y_der1 = runge_kutt_method(x, h, y_first, y_der_first)
    y2, y_der2, delta_y2, delta_y_der2 = runge_kutt_method(x1, 0.5*h, y_first, y_der_first)
    line2, = plt.plot(x, y1, label="Runge Kutt method")
    table = PrettyTable()
    print(" "*35, "Runge Kutt method")
    p = 4
    table._set_field_names(["k", "x", "y", "delta y", "delta y_der", "analitic y", "eps", "runge romberg precision"])
    for i in range(len(x)):
        if i == len(x) - 1:
            table.add_row([i, round(x[i], 6), round(y1[i], 6), delta_y1[i], delta_y_der1[i], round(f_y(x[i]), 6), round(abs(y1[i]-f_y(x[i])), 6), round(abs((y2[i*2]-y1[i])/(2**p-1)), 6)])
            continue
        table.add_row([i, round(x[i], 6), round(y1[i], 6), round(delta_y1[i], 6), round(delta_y_der1[i], 6), round(f_y(x[i]), 6), round(abs(y1[i]-f_y(x[i])), 6), round(abs((y2[i*2]-y1[i])/(2**p-1)), 6)])
    print(table)
    
    y1 = adams_method(a, b, y_first, y_der_first, h)
    y2 = adams_method(a, b, y_first, y_der_first, 0.5*h)
    print(" " * 25, "Adams method")
    p = 2
    table = PrettyTable()
    table._set_field_names(["k", "x", "y", "analitic y", "eps", "runge romberg precision"])
    for i in range(len(x)):
        table.add_row([i, x[i].round(6), y1[i].round(6), f_y(x[i]).round(6), round(abs(y1[i] - f_y(x[i])), 6), round(abs(y2[2*i]-y1[i])/(2**p-1), 6)])
    print(table)
    print()
    line3, = plt.plot(x, y1, label="Adams method")

    right_y = list(map(f_y, x))
    line4, = plt.plot(x, right_y, label="(x^2 + 1/x) * exp(x^0.5)")
    
    plt.legend(handles=[line1, line2, line3, line4])
    plt.show()