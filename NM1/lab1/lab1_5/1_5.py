import numpy as np
np.set_printoptions(precision=6)
import sys
from scipy.linalg import norm

if len(sys.argv) < 4:
    print('usage: python 1_5.py <input.txt> <output.txt> <log.txt>')
    sys.exit(1)
    
filename = sys.argv[1]
outfilename = sys.argv[2]
logfilename = sys.argv[3]


#filename = "1_5.txt"

try:
    with open(filename) as f:
        order = int(f.readline())
        mat_A = []
        for i in range(order):
            mat_A.append(list(map(float, f.readline().split())))
        mat_A = np.array(mat_A)
        my_precision = float(f.readline())
except IOError:
    print('Problem with file')
    sys.exit(1)
except ValueError:
    print('Incorrect file format')
    sys.exit(1)

with open(logfilename, 'wt') as f:
    print('Source matrix:\n', file=f)
    f.write(np.array2string(mat_A, separator=', '))
    print('\n', file=f)
    print('Specified precision:', my_precision, file=f)
    print(file=f)

def householder_transform(a, j):
    v = np.zeros(order)
    v[j] = a[j] + np.sign(a[j]) * norm(a[j:])
    v[j+1:] = a[j+1:]
    v = v[:, np.newaxis]
    #print("v", v)
    h = np.eye(order) - (2 / (v.T @ v)) * (v @ v.T)
    #print("H:", h)

    with open(logfilename, 'at') as f:
        print('Matrix after Householder transform:\n', h, file=f)
        print(file=f)

    return h

def qr_decomposition(r):
    q = np.eye(order)
    for i in range(order - 1):
        h = householder_transform(r[:, i], i)
        q = q @ h
        r = h @ r
    return q, r

def qr_algorithm():
    a = mat_A.copy()
    iter_log = 0
    while True:
        q, r = qr_decomposition(a)
        a = r @ q
        #print("q r a:", q, r, a, sep='\n')
        is_eigen = [True]*order
        is_real  = [True]*order
        j = 0
        while j < order - 1:
            if norm(a[j + 1:, j]) <= my_precision:
                is_eigen[j] = True
                is_real[j] = True
            elif j > order - 3 or norm(a[j + 2:, j]) <= my_precision:
                is_eigen[j] = True
                is_real[j] = False
                j += 1
            else:
                is_eigen[j] = False
            j += 1
        iter_log += 1
        with open(logfilename, 'at') as f:
            print('Iter i: ', iter_log, file=f)
            print('Result of qr_decomposition:\nQ: ', file=f)
            f.write(np.array2string(q, separator=', '))
            print(file=f)
            print('R: ', file=f)
            f.write(np.array2string(r, separator=', '))
            print('\n\nMatrix A_i (R @ Q):\n', file=f)
            f.write(np.array2string(a, separator=', '))
            print(file=f)
        if np.all(is_eigen):
            return compute_eigen(a, is_real)

def compute_eigen(matrix, is_real):
    sol = []
    j = 0
    while j < order:
        if is_real[j]:
            sol.append(matrix[j, j])
            j += 1
        elif j < order:
            a11 = matrix[j, j]
            a12 = matrix[j, j + 1]
            a21 = matrix[j + 1, j]
            a22 = matrix[j + 1, j + 1]
            sol.extend(np.roots((1, -a11 - a22, a11 * a22 - a12 * a21)))
            j += 2
    return np.array(sol)

with open(outfilename, 'wt') as f:
    print('Eigenvalues of the source matrix: ', file=f)
    f.write(np.array2string(qr_algorithm(), separator=', '))
