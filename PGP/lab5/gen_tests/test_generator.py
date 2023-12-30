import sys
import numpy as np
import random
from tqdm import tqdm

filename = sys.argv[1]
n = int(sys.argv[2])

with open(filename + '.in.data', 'w+b') as f:
	f.write(n.to_bytes(4, "little"))
	for i in tqdm(range(n)):
		f.write(random.randint(2, 6).to_bytes(1))
