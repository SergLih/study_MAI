import numpy as np
np.set_printoptions(precision=3)
import scipy.linalg 
import sys

if len(sys.argv) < 4:
    print('usage: python LU.py <input.txt> <output.txt> <log.txt>')
    sys.exit(1)

filename = sys.argv[1]
outfilename = sys.argv[2]
logfilename = sys.argv[3]

try:
    with open(filename) as f:
        order = int(f.readline())
        a = []
        for i in range(order):
            a.append(list(map(float, f.readline().split())))
        b = list(map(float, f.readline().split()))
        
except IOError:
    print('Problem with file')
    sys.exit(1)
except ValueError:
    print('Incorrect file format')
    sys.exit(1)

#temp = np.array(a)
a = np.array(a)
b = np.array(b)

l = np.zeros_like(a)     
u = np.zeros_like(a)
p = np.zeros_like(b)

for i in range(0, order):
    p[i] = i
for k in range(0, order):
    pi = 0
    for i in range(k, order):
        if abs(a[i, k]) > pi:
            pi = abs(a[i, k])
            k_new = i
    if pi == 0:
        print('This matrix is degenerate')
        sys.exit(1)
    p[k], p[k_new] = p[k_new].copy(), p[k].copy()
    for i in range(0, order):
        a[k, i], a[k_new, i] = a[k_new, i].copy(), a[k, i].copy()
    for i in range(k + 1, order):
        a[i, k] = a[i, k] / a[k, k]
        for j in range(k + 1, order):
            a[i, j] = a[i, j] - a[i, k] * a[k, j]

for i in range(0, order):
    for j in range(0, order):
        if i <= j:
            u[i, j] = a[i, j]
        else:
            l[i, j] = a[i, j]
    l[i, i] = 1

#print(a)
#print(p)

def solve_by_lu(b):
    z = np.zeros_like(b)
    z[0] = b[0]
    for i in range(1, order):
        z[i] = b[i] - (l[i, :i] * z[:i]).sum()

    x = np.zeros_like(z)
    x[-1] = z[-1] / u[-1, -1]
    for i in range(order - 2, -1, -1):
        x[i] = (z[i] - (u[i, i+1:] * x[i+1:]).sum()) / u[i, i]
    return x

for i in range(1, order):
    determinant = u[0][0]
    determinant *= u[i][i]

inv_mat = np.transpose(np.array([solve_by_lu(i) for i in np.eye(order)]))    
    
with open(logfilename, 'wt') as f_log:
    print("Source matrix:\n", file=f_log)
    f_log.write(np.array2string(a, separator=', '))
    print("\n\nMatrix L:\n", file=f_log)
    f_log.write(np.array2string(l, separator=', '))
    print("\n\nMatrix U:\n", file=f_log)
    f_log.write(np.array2string(u, separator=', '))
    print("\nCheck:\n", file=f_log)
    f_log.write(np.array2string(l@u, separator=', '))
    print("\n\nPermutation matrix:\n", file=f_log)
    f_log.write(np.array2string(p, separator=', '))
    
with open(outfilename, 'wt') as f_out:
    print("Results:\n", file=f_out)
    f_out.write(np.array2string(solve_by_lu(b), separator=', '))
    print("\n\nDeterminant of matrix A:\n", file=f_out)
    print(determinant, file=f_out)
    print("\nInverse matrix:\n", file=f_out)
    f_out.write(np.array2string(inv_mat, separator=', '))
