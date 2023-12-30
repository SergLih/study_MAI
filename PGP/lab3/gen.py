import sys
import numpy as np
from tqdm import tqdm

def change(x, noise = 20):  # 0...255
	delta = np.random.randint(-noise, noise)
	return min(max(x + delta, 0), 255)   # 0..255

def change3(x): #  [100, 200, 255, 3] --> [110, 198, 253, 3] 
	return [change(x[0]), change(x[1]), change(x[2]), x[3]]

def remove_cluster(x):  # [100, 200, 255, 3] --> [100, 200, 255, 0]
	return x[:3] + [0]


filename = sys.argv[1]
w = int(sys.argv[2])
h = int(sys.argv[3])
k = int(sys.argv[4])

initial_clusters = [[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255), i] for i in range(k)]

with open(filename + '.in.data', 'w+b') as f1:
	with open(filename + '.ans.data', 'w+b') as f2:
		with open(filename + '.test', 'w') as f3:

			f1.write(w.to_bytes(4, "little"))
			f1.write(h.to_bytes(4, "little"))
	#		f2.write(w.to_bytes(4, "little"))
	#		f2.write(h.to_bytes(4, "little"))

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
					if j % 10000 == 0:
						new_point = remove_cluster(change3(initial_clusters[i]))
					f1.write(bytearray(new_point))
					f2.write(bytearray(new_point))
					#100 * 20, k= 3
					# 2000 // 3 == 667, 667 ;  666

					z += 1
				z += 1



	# byte_arr = [120, 3, 255, 0, 100]
	# binary_format = bytearray(byte_arr)
	# f.write(binary_format)
