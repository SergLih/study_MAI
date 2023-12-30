#!/usr/bin/python

from random import *

fname = input('filename:           ')
v = int(input('number of vertices: '))
p = int(input('graph density (%):  '))/100



if p==1:
    d = [[randint(v+1, 2*v) for j in range(v)] for i in range(v)]
else:
    d = [[0]*v for i in range(v)]
    for i in range(int(v*v*p)):
        d[randint(0, v-1)][randint(0, v-1)] = randint(v+1, 2*v)

# shortest path:
for i in range(v-1):
    d[i][i+1] = 1
    
with open(fname, 'w') as f:
    print(v, file=f)
    for row in d:
        print(*row, file=f)
    print(0, v-1, file=f)   
    
