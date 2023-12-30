import sys
import random
import numpy as np

if len(sys.argv) < 3:
    print('specify number of equations, filename_in')
    sys.exit(1)

n = int(sys.argv[1])
file_in = sys.argv[2]

if n > 1000:
	random_arr = np.diag(np.random.choice([1, 0.5, 0.25, 2, 4], n))
else:
	random_arr = np.random.random((n, n))
# print(random_arr)
np.savetxt(file_in, random_arr, header=str(n), comments='')


# with open(file_in) as f_out:
#     lines = [line.split() for line in f_out]
# lines = np.array(lines, dtype='float')
print("Determinant from numpy: ", np.linalg.det(random_arr))

