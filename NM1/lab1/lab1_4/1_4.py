import numpy as np
np.set_printoptions(precision=6)
import scipy
import sys


if len(sys.argv) < 4:
    print('usage: python 1_4.py <input.txt> <output.txt> <log.txt>')
    sys.exit(1)
    
filename = sys.argv[1]
outfilename = sys.argv[2]
logfilename = sys.argv[3]

#filename = "1_4.txt"
#filename = "test_from_methodic.txt"

try:
    with open(filename) as f:
        order = int(f.readline())
        mat = []
        for i in range(order):
            mat.append(list(map(float, f.readline().split())))
        my_precision = float(f.readline())
except IOError:
    print('Problem with file')
    sys.exit(1)
except ValueError:
    print('Incorrect file format')
    sys.exit(1)

#Jacobi rotation method

with open(logfilename, 'wt') as f:
    print('Source matrix:', mat, file=f)
    print('Order: ', order, file=f)
    print(file=f)

u = np.eye(order)
k = 0
mat_new = mat.copy()
while True:
    u_new = np.eye(order)
    i_max = 0
    j_max = 0
    max_elem_matrix = -np.inf
    for i in range(order):
        for j in range(i + 1, order):
            if abs(mat_new[i][j]) > max_elem_matrix:
                i_max = i
                j_max = j
                max_elem_matrix = abs(mat_new[i][j])
    phi = np.pi / 4
    if mat_new[i_max][i_max] != mat_new[j_max][j_max]:
        phi = np.arctan(2 * mat_new[i_max][j_max] / (mat_new[i_max][i_max] - mat_new[j_max][j_max])) / 2

    u_new[i_max, j_max] = -np.sin(phi)
    u_new[j_max, i_max] = np.sin(phi)
    u_new[i_max, i_max] = np.cos(phi)
    u_new[j_max, j_max] = np.cos(phi)
    mat_new = np.transpose(u_new) @ mat_new @ u_new
    k += 1
    u = u @ u_new
    small_sum = np.sqrt(sum([mat_new[i][j] ** 2 for i in range(order) for j in range(i + 1, order)]))
    with open(logfilename, 'at') as f:
        print('Iter {0}: small_sum = {1} and precision = {2}'.format(k, small_sum, my_precision), file=f)
        print('i_max = {0} and j_max = {1}'.format(i_max, j_max), file=f)
        print('Matrix U:', file=f)
        f.write(np.array2string(u, separator=', '))
        print(file=f)
        print('Matrix A:', file=f)
        f.write(np.array2string(mat_new, separator=', '))
        print(file=f)
        print(file=f)
    if small_sum < my_precision:
        break
l = np.diag(mat_new)
#print(u)
#print(k)
#print(l)