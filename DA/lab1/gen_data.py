#!/usr/bin/python

from random import *

template = '+{}-{}-{}\t{}\n'
filename = input()
k = int(input())

with open(filename, 'w') as f:
    for i in range(k):
        f.write(template.format(randint(1, 999), randint(1, 999), randint(1, 10**7-1), randint(0, 2**64-1)))
       
