import numpy as np
from random import randint
from scipy.sparse import rand
import sys
import scipy.sparse as sp 
from scipy.sparse import diags
from numpy.linalg import norm, inv
from scipy.sparse.linalg import spilu
from time import time
np.set_printoptions(linewidth=200)

if len(sys.argv) < 5:
    print('usage: python course_work_bicg.py <input.txt> <output.txt> <log.txt> eps')
    sys.exit(1) 

filename    = sys.argv[1]
outfilename = sys.argv[2] 
logfilename = sys.argv[3]
EPS         = float(sys.argv[4])
'''
shape = randint(13, 15)
while True:
    matrix = np.random.random((shape, shape+1))
    m = matrix[:,:-1]
    m[m < 0.8] = 0
    if np.linalg.det(m) != 0:
        break

#print(matrix)
np.savetxt(filename, matrix)#, fmt='%3d')

mat2 = np.loadtxt(filename)
'''
#with open(logfilename, 'wt') as f_log:
#    print("Order matrix: ", len(mat2), file=f_log)
#    print("Source matrix:\n", mat2[:, :-1].round(1), file=f_log)
#    print("Answer's vector:\n", mat2[:, -1].round(1), file=f_log)
#   print("Precision: ", EPS, file=f_log)
with open(filename) as f:
    shape = int(f.readline())
    matrix = [[float(num) for num in line.split()]
              for _, line in zip(range(shape), f)]
    matrix[0].insert(0, 0)
    matrix[-1].append(0)
    a, b, c = zip(*matrix)
    matrix = diags([a[1:], b, c[:-1]], [-1, 0, 1])
    #print(matrix)
    matrix = sp.csc_matrix(matrix)

    b = np.array([float(num) for num in f.readline().split()])
    #return matrix, b    
    
#slau = mat2[:, :-1], mat2[:, -1]

class BiCG:
    def __init__(self, matrix, b, eps=EPS, max_iter = 2000):
        self.matrix = matrix#sp.csc_matrix(slau[0])
        self.b = b#slau[1]
        self.eps = eps
        self.order = self.matrix.shape[0]
        self.max_iter = max_iter

    def BiCGStab(self):
        with open(logfilename, 'at') as f_log:
            print("\nBiCGStab: ", file=f_log)
        x = np.zeros_like(self.b)
        r = self.b - self.matrix @ x
        r1 = r.copy()
        rho, alpha, omega = 1, 1, 1
        v, p = np.zeros_like(self.b), np.zeros_like(self.b)
        k = 0
        assert r @ r1 != 0
        while True:
            rho_prev, x_prev = rho, x.copy()
            rho = r1 @ r
            beta = (rho / rho_prev) * ( alpha / omega)
            p = r + beta * (p - omega * v)
            v = self.matrix @ p
            alpha = rho / (r1 @ v)
            s = r - alpha * v
            t = self.matrix @ s
            omega = np.dot(t, s) / np.dot(t, t)
            x += omega * s + alpha * p
            r = s - omega * t
            k += 1
            #if k % 1000 == 0:
            with open(logfilename, 'at') as f_log:
                print('Iter: {0:6d} ||x-x_prev|| = {1:>14.8f}\t ||r||={2:>14.8f}'.format(k, norm(x - x_prev), norm(r)), file=f_log)
            if norm(r) < self.eps:
                break
        return x
    
    def bicg(self):
        with open(logfilename, 'at') as f_log:
            print("\nBiCG: ", file=f_log)
        x = np.ones_like(self.b)
        r = self.b - self.matrix @ x
        p, z, s = r.copy(), r.copy(), r.copy()
        k = 0
        while k < self.max_iter:
            x_prev, p_prev, r_prev = x.copy(), p.copy(), r.copy()
            alpha = np.dot(p, r) / np.dot(s, (self.matrix @ z))
            x += alpha * z
            r -= alpha * self.matrix @ z
            p -= alpha * self.matrix.T @ s
            beta = np.dot(p, r) / np.dot(p_prev, r_prev)
            k += 1
            with open(logfilename, 'at') as f_log:
                print('Iter: {0:6d} ||x-x_prev|| = {1:>14.8f}\t ||r||={2:>14.8f}'.format(k, norm(x - x_prev), norm(r)), file=f_log)
            if norm(x - x_prev) < self.eps or norm(r) < self.eps:
                break
            z = r + beta*z
            s = p + beta*s
        return x
    
    def precond_bicg(self):
        with open(logfilename, 'at') as f_log:
            print("\nPrecond BiCGStab: ", file=f_log)
        M = spilu(self.matrix)
        x = np.zeros_like(self.b)#slau[1])
        r = self.b - self.matrix @ x
        r1, p = r, r
        k = 0
        #assert r @ r1 != 0
        if r @ r1 == 0:
            print("UUPS")
            sys.exit()
        while True:
            r_prev, x_prev = r.copy(), x.copy()
            p1 = M.solve(p)
            Ap = self.matrix @ p1
            alpha = (r @ r1) / (Ap @ r1)
            s = r - alpha * Ap
            if norm(s) < self.eps:
                k += 1
                x += alpha * p1
                with open(logfilename, 'at') as f_log:
                    print('Iter: {0:6d} ||x-x_prev|| = {1:>14.8f}\t ||r||={2:>14.8f}'.format(k, norm(x - x_prev), norm(r)), file=f_log)
                break
            z = M.solve(s)
            Az = self.matrix @ z
            omega = (Az @ s) / (Az @ Az)
            x += alpha * p1 + omega * z
            r = s - omega * Az
            beta = (r @ r1) / (r_prev @ r1) * (alpha / omega)
            p = r + beta * (p - omega * Ap)
            k += 1
            with open(logfilename, 'at') as f_log:
                print('Iter: {0:6d} ||x-x_prev|| = {1:>14.8f}\t ||r||={2:>14.8f}'.format(k, norm(x - x_prev), norm(r)), file=f_log)
            if norm(r) < self.eps:
                break
        return x

with open(outfilename, 'wt') as f_out:
    #print("Order matrix: ", len(matrix), file=f_out)#len(mat2), file=f_out)
    #print("Source matrix:\n", mat2[:, :-1].round(1), file=f_out)
    #print("Answer's vector:\n", mat2[:, -1].round(1), file=f_out)
    print("Precision: ", EPS, file=f_out)
solver = BiCG(matrix, b, max_iter=100000)#BiCG(slau, max_iter=100000)
'''
start0 = time()   
solver = BiCG(matrix, b, max_iter=100000)#BiCG(slau, max_iter=100000)
x0 = solver.bicg()
end0 = time()
with open(outfilename, 'at') as f_out:
    print("\nBiCG: \nx: ", x0, file=f_out)#str(x0[:10])[:-1], '... ]')
    print("Time: ", round(end0 - start0, 5), "sec", file=f_out)
'''

start1 = time()
x1 = solver.BiCGStab()
end1 = time()
with open(outfilename, 'at') as f_out:
    print("\nBiCGStab: \nx: ", x1, file=f_out)
    print("Time: ", round(end1 - start1, 5), "sec", file=f_out)

start2 = time()
x2 = solver.precond_bicg()
end2 = time()
with open(outfilename, 'at') as f_out:
    print("\nPrecond BiCGStab: \nx: ", x2, file=f_out)
    print("Time: ", round(end2 - start2, 5), "sec", file=f_out)


start3 = time()
#slau = mat2[:, :-1], mat2[:, -1]
x3 = np.linalg.solve(matrix.toarray(), b)#(slau[0], slau[1])
end3 = time()
with open(outfilename, 'at') as f_out:        
    print("\nBiCGStab from linalg: \nx: ", x3, file=f_out)#str(x3[:10])[:-1], '... ]')
    print("Time: ", round(end3 - start3, 5), "sec", file=f_out)

