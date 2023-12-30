import numpy as np
np.set_printoptions(precision=3)
import sys

if len(sys.argv) < 4:
    print('usage: python Running.py <input.txt> <output.txt> <log.txt>')
    sys.exit(1)

filename = sys.argv[1]
outfilename = sys.argv[2]
logfilename = sys.argv[3]

def get_data(filename):
    try:
        with open(filename) as f:
            order = int(f.readline())
            mat = []
            for i in range(order):
                mat.append(list(map(float, f.readline().split())))
            vec_ans = list(map(float, f.readline().split()))

            mat[0].insert(0, 0)
            mat[-1].append(0)
            #print("Source matrix:")
            #print(mat)

    except IOError:
        print('Problem with file')
        sys.exit(1)
    except ValueError:
        print('Incorrect file format')
        sys.exit(1)
    return mat, vec_ans

mat, vec_ans = get_data(filename)
order = len(mat)
a, b, c = zip(*mat)
p = [-c[0] / b[0]]
q = [vec_ans[0] / b[0]]
x = [0] * (order + 1)
for i in range(1, order):
    p.append(-c[i] / (b[i] + a[i] * p[i - 1]))
    q.append((vec_ans[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1]))
for i in reversed(range(order)):
    x[i] = p[i] * x[i + 1] + q[i]
x.pop(-1)

with open(logfilename, 'wt') as f_log:
    print("P:\n", file=f_log)
    print(p, file=f_log)
    print("\nQ:\n", file=f_log)
    print(q, file=f_log)

with open(outfilename, 'wt') as f_out:
    print("Result:\n", file=f_out)
    print(x, file=f_out)