import numpy as np
import sys

if len(sys.argv) < 3:
    print('usage: python 3_4.py <input.txt> <output.txt>')
    sys.exit(1)
    
filename = sys.argv[1]
outfilename = sys.argv[2]

def get_data(filename):
    points = []
    values = []
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
    
def find_interval(x, x_pres):
    for i in range(len(x)):
        if x[i] <= x_pres and x_pres <= x[i+1]:
            return i

def first_derivative(x, y, x_pres):
    i = find_interval(x, x_pres)
    finite_difference = (y[i+1] - y[i]) / (x[i+1] - x[i])
    finite_difference2 = (((y[i+2] - y[i+1]) / (x[i+2] - x[i+1])) - finite_difference) / (x[i+2] - x[i])
    return finite_difference + finite_difference2 * (2*x_pres - x[i] - x[i+1])
    
    
def second_derivative(x, y, x_pres):
    i = find_interval(x, x_pres)
    finite_difference  = (y[i+2] - y[i+1]) / (x[i+2] - x[i+1])
    finite_difference2 = (y[i+1] - y[i]) / (x[i+1] - x[i])
    return 2*(finite_difference - finite_difference2) / (x[i+2] - x[i])
   

with open(outfilename, 'wt') as f_out:
    print('Preset x: ', preset_x, file=f_out)
    print('x: ', points, file=f_out)
    print('f(x): ', values, file=f_out)
    print("\nFirst derivative: ", first_derivative(points, values, preset_x), file=f_out)
    print("\nSecond derivative: ", second_derivative(points, values, preset_x), file=f_out)