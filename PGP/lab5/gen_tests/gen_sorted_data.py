import sys
import numpy as np
from tqdm import tqdm

filename = sys.argv[1]
k = int(sys.argv[2])
a = np.repeat(7, 100000000)#np.random.randint(0, 255, k)

with open(filename + '.in.data', 'w+b') as f1:
	with open(filename + '.ans.data', 'w+b') as f2:
		f1.write(len(a).to_bytes(4, "little"))
		f1.write(bytearray(list(a)))
		a.sort()
		f2.write(bytearray(list(a)))
