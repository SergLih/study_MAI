import random
n = int(input("Array size: "))

with open("test" + str(n), "w") as f:
    print(n, file=f)
    for i in range(n):
        print(random.randint(1, 1000), file=f)
