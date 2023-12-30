import numpy as np
import math
import operator
from math import *
from functools import reduce
import sys
#filename =

if len(sys.argv) < 4:
    print('usage: python 3_1.py <input.txt> <output.txt> <log.txt>')
    sys.exit(1)

filename = sys.argv[1]
outfilename = sys.argv[2]
logfilename = sys.argv[3]

def get_data(filename):
    try:
        with open(filename) as f:
            x1 = list(map(float, f.readline().split()))
            x2 = list(map(float, f.readline().split()))
            preset_x = float(f.readline())
    except IOError:
        print('Problem with file')
        sys.exit(1)
    except ValueError:
        print('Incorrect file format')
        sys.exit(1)
    return x1, x2, preset_x

x1, x2, preset_x = get_data(filename)

#print('Preset x: ', preset_x, file=f_out)
#print('x: ', points, file=f_out)
#print('f(x): ', values, file=f_out)

def f(x):
    return acos(x)
    
def omega(x_values, x, i):
    return reduce(operator.mul, [x - x_values[j] for j in range(len(x_values)) if i != j])
      
def lagrange_interpolation(x, x_values):
    res = 0
    with open(logfilename, 'at') as f_log:
        print('\nLagrange\'s polynom: \n', file=f_log)
    for i in range(len(x_values)):
        f_w = f(x_values[i]) / omega(x_values, x_values[i], i)
        res += f_w * omega(x_values, x, i)
        with open(logfilename, 'at') as f_log:
            print('i: ', i, '\tx_i: ', x_values[i], '\tf_i: ', f(x_values[i]), '\tw_4(x_i): ', omega(x_values, x_values[i], i), '\tf_i/w_4(x_i): ', omega(x_values, x_values[i], i), 'x* - x_i: ', x - x_values[i], file=f_log)
    with open(outfilename, 'at') as f_out:
        print("\nInfelicity: ", abs(f(x) - res), file=f_out)
    return res
    #print("Lagrange: ", res)
    #print("infelicity: ", abs(f(x) - res))
    
def newton_interpolation(x, x_values):
    func = [f(i) for i in x_values]
    coefs = [func[i] for i in range(len(x_values))]
    with open(logfilename, 'at') as f_log:
        print('\nNewton\'s polynom: \n', file=f_log)
    for j in range(1, len(x_values)):
        for i in range(len(x_values) - 1, j - 1, -1):
            coefs[i] = float(coefs[i] - coefs[i - 1]) / float(x_values[i] - x_values[i - j])
        with open(logfilename, 'at') as f_log:
            print('i: ', j, '\tx_i: ', x_values[j], '\tf(x_i): ', func[j], file=f_log)
            print( 'f(x_0), f(x_i, x_i+1), f(x_i, x_i+1, x_i+2), f(x_0, x_1, x_2, x_3): ', coefs, file=f_log)
        #with open(logfilename, 'at') as f_log:
        #    print('f(): ', j, '\tx_i: ', x_values[j], file=f_log)
    #with open(logfilename, 'at') as f_log:
    #    print(, file=f_log)
    res = coefs[-1]
    
    for i in range(len(coefs) - 2, -1, -1):
        res = res * (x - x_values[i]) + coefs[i]
    with open(outfilename, 'at') as f_out:
        print("\nInfelicity: ", abs(f(x) - res), file=f_out)
    return res
        
with open(outfilename, 'wt') as f_out:
    print('Points: ', x1, file=f_out)
    with open(logfilename, 'wt') as f_log:
        print('\nPoints: ', x1, file=f_log)
    print("Lagrange's polynom: ", lagrange_interpolation(preset_x, x1), file=f_out)
    print("Newton's polynom: ", newton_interpolation(preset_x, x1), file=f_out)
    print(file=f_out)
    print('\nPoints: ', x2, file=f_out)
    with open(logfilename, 'at') as f_log:
        print('\nPoints: ', x2, file=f_log)
    print("Lagrange's polynom: ", lagrange_interpolation(preset_x, x2), file=f_out)
    print("Newton's polynom: ", newton_interpolation(preset_x, x2), file=f_out)