import numpy as np
np.set_printoptions(precision=6)
import scipy
import sys
from scipy.linalg import norm, inv

if len(sys.argv) < 4:
    print('usage: python 1_3.py <input.txt> <output.txt> <log.txt>')
    sys.exit(1)
    
filename = sys.argv[1]
outfilename = sys.argv[2]
logfilename = sys.argv[3]

#filename = "iter_zeid.txt"  # sys.argv[1]

try:
    with open(filename) as f:
        order = int(f.readline())
        mat = []
        for i in range(order):
            mat.append(list(map(float, f.readline().split())))
        vec_ans = list(map(float, f.readline().split()))
        my_precision = float(f.readline())
except IOError:
    print('Problem with file')
    sys.exit(1)
except ValueError:
    print('Incorrect file format')
    sys.exit(1)

beta = np.zeros_like(vec_ans)
alpha = np.zeros_like(mat)

#equivalent form

for i in range(order):
    beta[i] = vec_ans[i] / mat[i][i]
    for j in range(order):
        if i != j:
            alpha[i, j] = - mat[i][j] / mat[i][i]

with open(logfilename, 'wt') as f:
    print('Source matrix:', mat, file=f)
    print('Order: ', order, file=f)
    print('Vector answers: ', vec_ans, file=f)
    print('\nEquivalent form:', file=f)
    print("Alpha: ", file=f)
    f.write(np.array2string(alpha, separator=', '))
    print(file=f)
    print("Beta: ", file=f)
    f.write(np.array2string(beta, separator=', '))
    print(file=f)

#print("Beta: ", beta)
#print("Alpha: ", alpha)

def get_norm(alpha1=None, c=None):
    for i in range(order):
        tmp = 0
        #method Jacobi for equivalent form
        for j in range(1, order):
            if i != j:
                tmp += abs(mat[i][j])
        if abs(mat[i][i]) <= tmp:
            return lambda x_new, x: norm(x_new - x, float('inf'))
    if c is not None:
        coef = norm(c, float('inf')) / (1 - norm(alpha1, float('inf')))
    coef = norm(alpha, float('inf')) / (1 - norm(alpha, float('inf')))
    return lambda x_new, x: coef * norm(x_new - x, float('inf'))

x = np.zeros_like(beta)
x_new = np.zeros_like(beta)
x = beta
k = 0
my_norm = get_norm()

#Iter method
with open(logfilename, 'at') as f:
    print('\nSimple iteration method:\n', file=f)
while True:
    x_new = beta + np.dot(alpha, x)
    k += 1
    with open(logfilename, 'at') as f:
        print('Iter {0}: norm = {1} and precision = {2}'.format(k, my_norm(x_new, x), my_precision), file=f)
        print(file=f)
    if my_norm(x_new, x) <= my_precision:
        break
    x = x_new

#print(alpha)
#print(beta)
print(k)
print(x)
 
            
#Seidel method
with open(logfilename, 'at') as f:
    print('Seidel method:\n', file=f)

mat_b = np.zeros_like(alpha)
mat_c = np.zeros_like(alpha)

for i in range(0, order):
    for j in range(0, order):
        if i > j:
            mat_b[i, j] = alpha[i, j]
        else:
            mat_c[i, j] = alpha[i, j]

            
x = np.zeros_like(beta)
x_new = np.zeros_like(beta)
x = beta
k = 0
my_norm = get_norm()            

inv_mat = scipy.linalg.inv((np.eye(len(np.zeros_like(mat_b)))) - mat_b)            
#print(inv_mat)
while True:
    x_new = inv_mat @ (np.dot(mat_c, x)) + np.dot(inv_mat, beta)
    k += 1
    with open(logfilename, 'at') as f:
        print('Iter {0}: norm = {1} and precision = {2}'.format(k, my_norm(x_new, x), my_precision), file=f)
        print(file=f)
    if my_norm(x_new, x) <= my_precision:
        break
    x = x_new    

print(k)
print(x)
            
#print("Matrix B:", mat_b)
#print("Matrix_C:", mat_c)           

with open(outfilename, 'wt') as f:

    print('Seidel method\n', file=f)
    print('Solution: ', *x, sep='\n\t', file=f)
    print('Number of iterations: ', k, file=f)