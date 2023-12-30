#!/usr/bin/python

from random import *
from string import ascii_letters

template = '{key} {value}\n'

n = 100
with open('input2', 'w') as f:
    #f.write(str(n) + "\n")
    for i in range(1, n + 1):
        if i % 1000000 == 0:
            print(i)
        f.write(template.format(**{
            'key': ''.join(choice(ascii_letters) for _ in range(randint(1, 256))),
            'value': str(randint(1, 18446744073709551615))
        }))