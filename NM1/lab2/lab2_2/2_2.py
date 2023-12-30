import numpy as np
from numpy.linalg import norm
from math import *
import sys

interval_x1 = (8, 10)
interval_x2 = (2, 3)

if len(sys.argv) < 4:
    print('usage: python 2_2.py <eps> <output.txt> <log.txt>')
    sys.exit(1)
    
EPS = float(sys.argv[1])
outfilename = sys.argv[2]
logfilename = sys.argv[3]

def f1(x):
    return x[0]**2 + x[1]**2 - 16

def f2(x):
    return x[0] - exp(x[1]) + 4

def phi1(x):
    return (16 - x[1]**2)**0.5

def phi2(x):
    return log(x[0] + 4)

with open(logfilename, 'wt') as f_log:
    print('f1: x1^2 + x2^2 - 16 ', file=f_log)
    print('f2: x1 + exp(x2) + 16 ', file=f_log)
    print('\nPrecision: ', EPS, file=f_log)
    print('\nInterval x1: ', interval_x1,  file=f_log)
    print('\nInterval x2: ', interval_x2, file=f_log)
    print(file=f_log)
    
def derivative(x, f1=False, f2=False, phi1=False, phi2=False, x1=False, x2=False):
    if f1 and x1:
        return 2 * x[0]
    elif f1 and x2:
        return 2 * x[1]
    elif f2 and x1:
        return 1
    elif f2 and x2:
        return -exp(x[1])
    elif (phi1 and x1) or (phi2 and x2):
        return 0
    elif phi1 and x2:
        return -x[1] / ((16 - x[1]**2)**0.5)
    elif phi2 and x1:
        return 1 / (x[0] + 4)
        
def get_q(x):
    max_phi1 = (abs(derivative(x, phi1=True, x1=True)) + 
                abs(derivative(x, phi1=True, x2=True)))
    max_phi2 = (abs(derivative(x, phi2=True, x1=True)) + 
                abs(derivative(x, phi2=True, x2=True)))
    return max(max_phi1, max_phi2)
    
    
def A1(x):
    return [[f1(x), derivative(x, f1=True, x2=True)],
            [f2(x), derivative(x, f2=True, x2=True)]]

def A2(x):
    return [[derivative(x, f1=True, x1=True), f1(x)],
            [derivative(x, f2=True, x1=True), f2(x)]]
            
def jacobi(x):
    return [[derivative(x, f1=True, x1=True), derivative(x, f1=True, x2=True)],
            [derivative(x, f2=True, x1=True), derivative(x, f2=True, x2=True)]]
            
def determinant(mat):
    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

def newton_method():
    x = [(interval_x1[0] + interval_x1[1]) / 2, 
         (interval_x2[0] + interval_x2[1]) / 2]
    cnt_iter = 0
    with open(logfilename, 'at') as f_log:
        print('Newton method:\n ', file=f_log)
    while True:
        x_next = [x[0] - determinant(A1(x)) / determinant(jacobi(x)),
                  x[1] - determinant(A2(x)) / determinant(jacobi(x))]
        with open(logfilename, 'at') as f_log:
            print('Iter: ', cnt_iter, '\tx_1: ', x[0], '\tx_2: ', x[1], '\tf1(x1, x2): ', f1(x), '\tf2(x1, x2): ', f2(x), file=f_log)
            print('Der_f1_x1: ', derivative(x, f1=True, x1=True), '\tDer_f2_x1: ', derivative(x, f2=True, x1=True), '\tDer_f1_x2: ', derivative(x, f1=True, x2=True), '\tDer_f2_x2: ', derivative(x, f2=True, x2=True), file=f_log)
            print('Det A1: ', determinant(A1(x)), '\tDet A2: ', determinant(A2(x)), '\tDet Jacobi: ', determinant(jacobi(x)), file=f_log)
            print(file=f_log)
        cnt_iter += 1
        if max([abs(i - j) for i, j in zip(x, x_next)]) < EPS:
            return x_next
        
        x = x_next
        
def iteration_method():
    x = [(interval_x1[0] + interval_x1[1]) / 2, 
         (interval_x2[0] + interval_x2[1]) / 2]
    q = get_q(x)
    cnt_iter = 0
    with open(logfilename, 'at') as f_log:
        print('Iteration method:\nSelect q:  ', q, '\n', file=f_log)
    while True:
        x_next = [phi1(x), phi2(x)]
        with open(logfilename, 'at') as f_log:
            print('Iter: ', cnt_iter, '\tx_1: ', x[0], '\tx_2: ', x[1], '\tPhi1(x_1): ', phi1(x), '\tPhi2(x_2): ', phi2(x), file=f_log)
            print(file=f_log)
        cnt_iter += 1
        if max([abs(i - j) for i, j in zip(x, x_next)]) * q / (1 - q) < EPS:
            return x_next
        x = x_next
        
with open(outfilename, 'wt') as f_out:
    print("Newton method: ", newton_method(), file=f_out)
    print("\nIteration_method: ", iteration_method(), file=f_out)