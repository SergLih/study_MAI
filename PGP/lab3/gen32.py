import sys
import numpy as np
from tqdm import tqdm
import itertools

def change(x, noise = 15):  # 0...255
	new_x = int(np.random.normal(x, noise))
	return min(max(new_x, 0), 255)   # 0..255

def change3(x): #  [100, 200, 255, 3] --> [110, 198, 253, 3] 
	return [change(x[0]), change(x[1]), change(x[2]), x[3]]

def remove_cluster(x):  # [100, 200, 255, 3] --> [100, 200, 255, 0]
	return x[:3] + [0]


filename = sys.argv[1]
w = int(sys.argv[2])
h = int(sys.argv[3])
#k = int(sys.argv[4])
k = 8#32

r = [50, 90]#[0, 80, 160, 240]
g = [60, 200]#[0, 80, 160, 240]
b = [40, 240]

a = list(itertools.product(r, g, b))
initial_clusters = list(map(lambda x: [*x[1], x[0]], enumerate(a)))

with open(filename + '.in.data', 'w+b') as f1:
	with open(filename + '.ans.data', 'w+b') as f2:
		with open(filename + '.test', 'w') as f3:

			f1.write(w.to_bytes(4, "little"))
			f1.write(h.to_bytes(4, "little"))
			f2.write(w.to_bytes(4, "little"))
			f2.write(h.to_bytes(4, "little"))

			f3.write("{}.in.data\n{}.out.data\n{}\n".format(filename, filename, k))
			z = 0
			for i in range(k):
				if i < k - 1:
					n = w * h // k 
				else:
					n = w * h - z

				print("cluster {}: {} points, center: z={}, {} {}".format(i, n, z, z%w, z//w))
			
				f1.write(bytearray(remove_cluster(initial_clusters[i])))
				f2.write(bytearray(initial_clusters[i]))
				f3.write("{} {}\n".format(z%w, z//w))
				new_point = remove_cluster(change3(initial_clusters[i]))
				for j in tqdm(range(n-1)):
					#if j % 10000 == 0 or w * h < 10000:
					new_point = change3(initial_clusters[i])
					f1.write(bytearray(remove_cluster(new_point)))
					f2.write(bytearray(new_point))

					z += 1
				z += 1



	# byte_arr = [120, 3, 255, 0, 100]
	# binary_format = bytearray(byte_arr)
	# f.write(binary_format)
