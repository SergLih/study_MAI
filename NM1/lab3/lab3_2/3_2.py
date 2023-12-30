import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 4:
    print('usage: python 3_2.py <input.txt> <output.txt> <log.txt>')
    sys.exit(1)

filename = sys.argv[1]
outfilename = sys.argv[2]
logfilename = sys.argv[3]

def get_data(filename):
    try:
        with open(filename) as f:
            points = list(map(float, f.readline().split()))
            values = list(map(float, f.readline().split()))
            preset_x = float(f.readline())
    except IOError:
        print('Problem with file')
        sys.exit(1)
    except ValueError:
        print('Incorrect file format')
        sys.exit(1)
    return points, values, preset_x

points, values, preset_x = get_data(filename)

#progonka
def tma_running(matrix, vec_ans, order):
    a, b, c = zip(*matrix)
    p = [-c[0] / b[0]]
    q = [vec_ans[0] / b[0]]
    x = [0] * (order + 1)
    for i in range(1, order):
        p.append(-c[i] / (b[i] + a[i] * p[i - 1]))
        q.append((vec_ans[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1]))
    for i in reversed(range(order)):
        x[i] = p[i] * x[i + 1] + q[i]
    x.pop(-1)
    return x
    
def spline_interpolation(points, values, x):
    size = len(points)
    h = [points[i] - points[i - 1] for i in range(1, size)]
    mtrx = [[0, 2 * (h[0] + h[1]), h[1]]]                                             #a1b1 0 0
    b = [3 *((values[2] - values[1]) / h[1] - (values[1] - values[0]) / h[0])]        #c1a2b2 0
                                                                                      # 0c2a3b3
                                                                                      # 0 0c3a3

    for i in range(1, size - 3):
        mtrx.append([h[i], 2 * (h[i] + h[i + 1]), h[i + 1]])
        b.append(3 * ((values[i + 2] - values[i + 1]) / h[i + 1] - (values[i + 1] - values[i]) / h[i]))
    mtrx.append([h[-2], 2 * (h[-2] + h[-1]), 0])
    b.append(3 * ((values[-1] - values[-2]) / h[-1] - (values[-2] - values[-3]) / h[-2]))
    
    
    #find a, b, c, d
    
    c = tma_running(mtrx, b, size - 2)
    a = []
    b = []
    d = []
    c.insert(0, 0)
    with open(logfilename, 'wt') as f_log:
        print('X*: ', preset_x, file=f_log)
        print('x:', points, file=f_log)
        print('f(x):', values, file=f_log)
        print('Coefficients of interpolation:\n', file=f_log)
    for i in range(1, size):
        a.append(values[i - 1])
        with open(logfilename, 'at') as f_log:
            print('i: {0} \tinterval: [{1}, {2}]'.format(i, points[i-1], points[i]), file=f_log)
            print('\ta: ', a[i-1], file=f_log)
        if i < size - 1:
            d.append((c[i] - c[i - 1]) / (3 * h[i - 1]))
            b.append((values[i] - values[i - 1]) / h[i - 1] -
                     h[i - 1] * (c[i] + 2 * c[i - 1]) / 3)
            with open(logfilename, 'at') as f_log:
                print('\tb: ', b[i-1], '\n\tc: ', c[i-1], '\n\td: ', d[i-1], file=f_log)
    b.append((values[-1] - values[-2]) / h[-1] - 2 * h[-1] * c[-1] / 3)
    d.append(-c[-1] / (3 * h[-1]))
    with open(logfilename, 'at') as f_log:
        print('\tb: ', b[-1], '\n\tc: ', c[-1], '\n\td: ', d[-1], file=f_log)
    return a, b, c, d

coef = spline_interpolation(points, values, preset_x)    
    
def find_f_with_new_x(points, x):
    k = 0
    a, b, c, d = coef
    for i, j in zip(points, points[1:]):
        if i <= x <= j:
            with open(logfilename, 'at') as f_log:
                print('x_i-1 <= x* <= x_i: ', i, '<=', x, '<=', j, file=f_log)
            return a[k] + b[k] * (x - points[k]) + c[k]*((x-points[k])**2) + d[k]*((x-points[k])**3)
        k += 1
        
def draw_graphic(points):
    x1 = np.linspace(points[0], points[-1], 100)
    y1 = []
    a, b, c, d = coef
    for p in x1:
        k = 0
        for i, j in zip(points, points[1:]):
            if i <= p <= j:
                y1.append(a[k] + b[k] * (p - points[k]) + c[k]*((p-points[k])**2) + d[k]*((p-points[k])**3))
            k += 1

    plt.plot(x1, y1, color='b')
    plt.plot(points, values, 'or')
    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.legend(['y = x^3', 'y = 2x - x^2 + 1'], loc='upper left')
    plt.grid(True)
    plt.savefig('Spline interpolation')
    
with open(outfilename, 'wt') as f_out:
    print("\nS(x*): ", find_f_with_new_x(points, preset_x), file=f_out)
draw_graphic(points)