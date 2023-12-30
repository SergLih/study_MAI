import random
from tqdm import tqdm_notebook
from time import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np


class EllipticCurveGF:
    def __init__(self, a, b, p):
        assert (4*a**3 + 27*b**2) % p != 0
        self.a = a
        self.b = b
        self.p = p
        
    def __str__(self):
        return 'y^2 = x^3 + {}x + {} [{}]'.format(self.a, self.b, self.p)
    
    def __repr__(self):
        return self.__str__()
    
    def contains(self, pt):
        return (pt.y**2 - pt.x**3 - self.a * pt.x - self.b) % self.p == 0
    
    def points(self):
        for x in range(0, self.p):
            for y in range(0, self.p):
                pt = PointGF(x, y, self)
                if self.contains(pt):
                    yield pt
        yield PointGF(0, 0, self)   
        
class PointGF:
    def __init__(self, x, y, ec):
        self.ec = ec
        self.p = ec.p
        self.x = x
        self.y = y
        
    def __str__(self):
        return '({}, {}) [{}]'.format(self.x, self.y, self.p)
    
    def __repr__(self):
        return self.__str__()
    
    def zero(self):
        return PointGF(x=0, y=0, ec = self.ec)
    
    def __neg__(self):
        return PointGF(self.x, -self.y % self.p, self.ec)
    
    def __eq__(self, other):
        assert self.p == other.p
        return self.x == other.x and self.y == other.y
    
    def __add__(self, other):
        assert self.p == other.p
        Z  = self.zero()
        if self == Z:
            return other
        elif other == Z:
            return self
        elif self.x == other.x and self.y != other.y:
            return Z

        if self != other:
            m = ((self.y - other.y) * self._inv(self.x - other.x)) % self.p
        else:
            m = ((3 * self.x ** 2 + (ec.a % self.p)) * self._inv(2 * self.y)) % self.p
        xR = (m ** 2 - self.x - other.x) % self.p
        yR = (other.y + m * (xR - other.x)) % self.p
        return PointGF(xR, -yR % self.p, self.ec)
    
    def _inv(self, n):
        gcd, x, y = self._extended_euclidean_algorithm(n, self.p)
        assert (n * x + self.p * y) % self.p == gcd, 'n, p = {}, {}\ngcd, x, y = {}, {}, {}\n{} % {} == {}'.format(
            n, self.p, gcd, x, y, n * x + self.p * y, self.p, gcd)

        if gcd != 1:
            raise ValueError(
                '{} has no multiplicative inverse '
                'modulo {}'.format(n, self.p))
        else:
            return x % self.p
        
    def _extended_euclidean_algorithm(self, a, b):
        s, old_s = 0, 1
        t, old_t = 1, 0
        r, old_r = b, a

        while r != 0:
            quotient = old_r // r
            old_r, r = r, old_r - quotient * r
            old_s, s = s, old_s - quotient * s
            old_t, t = t, old_t - quotient * t

        return old_r, old_s, old_t
    
    def order(self):
        i = 1
        temp = self 
        while temp != self.zero():
            temp = self + temp 
            i += 1
        return i

def lowest_set_bit(a):
    b = (a & -a)
    low_bit = -1
    while (b):
        b >>= 1
        low_bit += 1
    return low_bit

def to_bits(k):
    k_binary = bin(k)[2:]
    return (bit == '1' for bit in k_binary[::-1])

def pow_mod(a, k, m):
    r = 1
    b = a
    for bit in to_bits(k):
        if bit:
            r = (r * b) % m
        b = (b * b) % m
    return r

def primality_test_miller_rabin(a, MILLER_RABIN_ITERATIONS=20):
    if a == 2:
        return True
    if a == 1 or a % 2 == 0:
        return False
    
    m = a - 1
    lb = lowest_set_bit(m)
    m >>= lb
    for _ in range(MILLER_RABIN_ITERATIONS):
        b = random.randint(2, a - 1)
        j = 0
        z = pow_mod(b, m, a)
        while not ((j == 0 and z == 1) or z == a - 1):
            if j > 0 and z == 1 or j + 1 == lb:
                return False
            j += 1
            z = (z * z) % a
    return True


def test_points(a, b, p_start, n_pts=100):
    res = pd.DataFrame(columns=['p', 't', 'order', 'pt'])
    if p_start % 2 == 0:
        p_start += 1
    pp = p_start
    while True:
        pp += 2
        p_is_prime = primality_test_miller_rabin(pp)
        if p_is_prime:
            try:
                ec = EllipticCurveGF(a, b, pp)
            except:
                continue
            print('\np = ', pp)
            pts = ec.points()
            pts_cnt = 0
            while pts_cnt < n_pts:
                pt = next(pts)
                start = time()
                try:
                    order = pt.order()
                except:
                    continue
                x = pp*random.uniform(0.9, 1.1)
                res = res.append({'p': pp, 'x': x, 'pt': pt, 't' : time() - start, 'order': order+0.0}, ignore_index=True)
                print('.', end='')
                pts_cnt += 1
            break
    return res


a=random.randint(1, 10**8)
b=random.randint(1, 10**8)
print(a, b)
res = pd.DataFrame()
for p in range(9, 18):
    res = res.append(test_points(a, b, 2**p + random.randint(10, 100), 40), ignore_index=True)


sns.set(font_scale=1.1)
plt.figure(figsize=(12, 6))
ax = sns.scatterplot(data=res, x='x', y='t', hue='order', palette='plasma')#,  palette=sns.light_palette("purple"))
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.set_xticks([10**3, 10**4])
ax.set_xlabel('p')
plt.xscale('log')
plt.show()    
 
 

res2 = test_points(a, b, int(6e7) + random.randint(10, 100), 30)

sns.set(font_scale=1.1)
plt.figure(figsize=(3, 6))
ax = sns.scatterplot(data=res2, x='x', y='t', hue='order', palette='plasma')#,  palette=sns.light_palette("purple"))
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xticks([])
ax.set_xlabel('p = 60000103')
# plt.xscale('log')
plt.show()        