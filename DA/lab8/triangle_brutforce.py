from itertools import *
from math import *


def triangle_exists(a, b, c):
    return a + b > c and a + c > b and b + c > a


def triangle_area(a, b, c):
    p = (a + b + c)/2
    return sqrt(p*(p-a)*(p-b)*(p-c))


# n = int(input())
# sides = [int(input()) for i in range(n)]

sides = [10, 99, 16, 24, 58, 29, 12, 46, 47, 4, 25, 50, 49]
print(sorted(sides))
areas = [(comb, triangle_area(*comb)) for comb in combinations(sides, 3)
                                      if(triangle_exists(*comb))]
print(areas)
print(max(areas, key=lambda x: x[1]))