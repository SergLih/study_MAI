import numpy as np
import sys

def f(x):
    return x / (x**2 + 9)

if len(sys.argv) < 4:
    print('usage: python 3_5.py <input.txt> <output.txt> <log.txt>')
    sys.exit(1)

filename = sys.argv[1]
outfilename = sys.argv[2]
logfilename = sys.argv[3]
    
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

points, var_h = get_data(filename)

with open(logfilename, 'wt') as f_log:
    print('Function: x / (x^2 + 9)', file=f_log)
    print('x_0: ', points[0], '\tx_k: ', points[1], file=f_log)
    print(file=f_log)
    print('\nh1: ', var_h[0], file=f_log)
    
def method_rectangle(x_0, x_k, h):
    x = np.arange(x_0, x_k + h, h)
    f_i = 0
    with open(logfilename, 'at') as f_log:
        print('Method rectangle:\n', file=f_log)
    for i in range(1, len(x)):
        f_i += f((x[i-1] + x[i])/2)
        with open(logfilename, 'at') as f_log:
            print('i: ', i, '\tx_i: ', x[i], '\tf_i: ', f_i, file=f_log)
    with open(logfilename, 'at') as f_log:
        print('\th*f_i: ', h*f_i, file=f_log)
    return h*f_i
    
def method_trapeze(x_0, x_k, h):
    x = np.arange(x_0, x_k + h, h)
    f_i = 0
    with open(logfilename, 'at') as f_log:
        print('\nMethod trapeze:\n', file=f_log)
    for i in range(1, len(x)):
        f_i += (f(x[i]) + f(x[i-1]))
        with open(logfilename, 'at') as f_log:
            print('i: ', i, '\tx_i: ', x[i], '\tf_i: ', f_i, file=f_log)
    with open(logfilename, 'at') as f_log:
        print('\t0.5*h*f_i: ', 0.5 * h * f_i, file=f_log)
    return 0.5 * h * f_i
    
def simpson_method(x_0, x_k, h):
    n = int((x_k - x_0) / h)
    f_i = f(x_0) + f(x_k)
    with open(logfilename, 'at') as f_log:
        print('\nSimpson\'s method:\n', file=f_log)
        print('i: ', 0, '\tf(x_0) + f(x_k): ', f_i, file=f_log)
    for i in range(1, n):
        if i % 2 != 0:
            f_i += 4 * f(x_0 + i * h)
        else:
            f_i += 2 * f(x_0 + i * h)
        with open(logfilename, 'at') as f_log:
            print('i: ', i, '\tf_i: ', f_i, file=f_log)
    with open(logfilename, 'at') as f_log:
        print('\th / 3 * f_i: ', h / 3 * f_i, file=f_log)
    return h / 3 * f_i    
    
    
def runge_romberg(F_h, F_kh, k, p):
    return (F_kh - F_h) / (k ** p - 1)


# print(rect) 
# print(runge_romberg(rect[0], rect[-1], (var_h[-1]/var_h[0]), 2))
# print(trapez)
# print(runge_romberg(trapez[0], trapez[-1], (var_h[-1]/var_h[0]), 2))
# print(simps)
# print(runge_romberg(simps[0], simps[-1], (var_h[-1]/var_h[0]), 2))

with open(outfilename, 'wt') as f_out:
    rect = [method_rectangle(points[0], points[1], var_h[0]), method_rectangle(points[0], points[1], var_h[1])]
    trapez = [method_trapeze(points[0], points[1], var_h[0]), method_trapeze(points[0], points[1], var_h[1])]
    simps = [simpson_method(points[0], points[1], var_h[0]), simpson_method(points[0], points[1], var_h[1])]
    print("Step(h): ", var_h[0], file=f_out)
    print("\nMethod rectangle: ", method_rectangle(points[0], points[1], var_h[0]), file=f_out)
    print("Method trapeze: ", method_trapeze(points[0], points[1], var_h[0]), file=f_out)
    print("Simpson's method: ", simpson_method(points[0], points[1], var_h[0]), file=f_out)
    print(file=f_out)
    with open(logfilename, 'at') as f_log:
        print('\nh2: ', var_h[1], file=f_log)
    print("Step(h): ", var_h[1], file=f_out)
    print("\nMethod rectangle: ", method_rectangle(points[0], points[1], var_h[1]), file=f_out)
    print("Method trapeze: ", method_trapeze(points[0], points[1], var_h[1]), file=f_out)
    print("Simpson's method: ", simpson_method(points[0], points[1], var_h[1]), file=f_out)