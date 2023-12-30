import itertools as it
import random

s1 = ''.join(random.choices('abABc\n',  k=random.randint(5, 20)))
s2 = ''.join(random.choices('abABc\n',  k=random.randint(5, 20)))
p = 0.4
options = ''
if random.random() < 0.5:
    options += 'i'
if random.random() < p:
    options += 'w'
if random.random() < p:
    options += 'B'
if random.random() < p:
    options += 'Z'

if options != '':
    options = '-' + options

fname1 = '1.txt'
fname2 = '2.txt'

with open(fname1, 'w') as f1:
    f1.write(s1)
with open(fname2, 'w') as f2:
    f2.write(s2)
with open('opts.txt', 'w') as f:
    f.write(options)
