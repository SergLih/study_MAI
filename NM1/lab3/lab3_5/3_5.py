import numpy as np
from prettytable import PrettyTable
import sys

if len(sys.argv) < 3:
    print('usage: python 3_5.py <input.txt> <output.txt>')
    sys.exit(1)

filename = sys.argv[1]
outfilename = sys.argv[2]

def get_data(filename):
    try:
        with open(filename) as f:
            points = list(map(float, f.readline().split()))
            var_h = list(map(float, f.readline().split()))
    except IOError:
        print('Problem with file')
        sys.exit(1)
    except ValueError:
        print('Incorrect file format')
        sys.exit(1)
    return points, var_h


def method_rectangle(x, h):
    res = 0
    results = [res]
    for i in range(1, len(x)):
        res += h*f((x[i] + x[i-1])/2)
        results.append(res)
    return results


def method_trapeze(x, h):
    res = 0
    results = [res]
    for i in range(1, len(x)):
        res += 0.5 * h * (f(x[i]) + f(x[i - 1]))
        results.append(res)
    return results


def simpson_method(x, h):
    res = 1/3*h*f(x[0])
    results = [res]
    for i in range(1, len(x)-1):
        if i % 2:
            res += 1/3*h*4*f(x[i])
        else:
            res += 1/3*h*2*f(x[i])
        results.append(res)
    res += 1/3*h*f(x[len(x)-1])
    results.append(res)
    return results


def runge_rumberg_method(h1_res, h2_res, h1, h2):
    results = []
    for i in range(len(h1_res)):
        if 0 <= i < 2: 
            results.append((h2_res[i] - h1_res[i]) / ((h1 / h2)**2 - 1))
        else:
            results.append((h2_res[i] - h1_res[i]) / ((h1 / h2)**4 - 1))
    return results


if __name__ == "__main__":
    f = lambda x: x / (x**2 + 9)
    points, var_h = get_data(filename)
    x_0 = points[0]
    x_k = points[-1]
    h1 = var_h[0]
    h2 = var_h[-1]
    x = np.arange(x_0, x_k + h1, h1)
    y = []
    for i in range(len(x)):
        y.append(f(x[i]))
    table = PrettyTable()
    table._set_field_names(["step", "x", "y", "rectangle method", "trapeze method", "simpson method"])
    h1_res = [method_rectangle(x, h1), method_trapeze(x, h1), simpson_method(x, h1)]
    for i in range(len(x)):
        table.add_row([i, x[i], y[i]] + [h1_res[j][i] for j in range(len(h1_res))])
    h1_res = np.array(h1_res)[:, len(x)-1]
    with open(outfilename, 'wt') as f_out:
        print(table, file=f_out)

    x = np.arange(x_0, x_k + h2, h2)
    y = []
    for i in range(len(x)):
        y.append(f(x[i]))
    table = PrettyTable()
    table._set_field_names(["step", "x", "y", "rectangle method", "trapeze method", "simpson method"])
    h2_res = [method_rectangle(x, h2), method_trapeze(x, h2), simpson_method(x, h2)]
    for i in range(len(x)):
        table.add_row([i, x[i], y[i]] + [h2_res[j][i] for j in range(len(h2_res))])
    h2_res = np.array(h2_res)[:, len(x) - 1]
    with open(outfilename, 'at') as f_out:
        print(table, file=f_out)
        print(file=f_out)
        print("\nRunge_rumberg_method\n", file=f_out)
    table = PrettyTable()
    table._set_field_names(["f(x)", "analitic value", "rec", "trap", "simps", "err rec method", "err trap method", "err simp method"])
    analitic_value = 0.18386
    vals_runge_rumb = runge_rumberg_method(h1_res, h2_res, h1, h2)
    table.add_row(["x/(x**2 + 9)", analitic_value, vals_runge_rumb[0], vals_runge_rumb[1], vals_runge_rumb[2], abs(vals_runge_rumb[0] - analitic_value).round(9),
                   abs(vals_runge_rumb[1] - analitic_value).round(9), abs(vals_runge_rumb[2] - analitic_value).round(9)])
    with open(outfilename, 'at') as f_out:
        print(table, file=f_out)