import random

n = int(input())

edges = []
for i in range(1, n):
    
    for j in range(i+2, n):
        if random.random() > 0.5:
            edges.append((i, j, random.randint(100000, 100000000)))
    # edges.append((i, i+1, 10))

print("{} {} {} {}".format(n, len(edges), 1, n))
for edge in edges:
    print(*edge)
#print(1, n, 1)
