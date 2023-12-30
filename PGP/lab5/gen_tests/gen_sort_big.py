import sys
import numpy as np
from tqdm import tqdm
from collections import *

filename = sys.argv[1]
# a = np.array(list(range(1, 256)) + [1])

n = 536870912
n_a = 1024**2
a = np.random.randint(0, 256, n_a)
c = Counter(a)
for i in range(256):
	print(i, ":", c[i], end='\t')

with open(filename + '.in.data', 'w+b') as f1:
	f1.write(n.to_bytes(4, "little"))
	for i in tqdm(range(n//n_a)):
		np.random.shuffle(a)
		f1.write(bytearray(list(a)))
		
with open(filename + '.ans.data', 'w+b') as f2:
    for i in tqdm(range(0, 256)):
        f2.write(bytearray([i]*(c[i]*(n//n_a))))
