import string
import random
import sys

def random_string(length = 256):
    return ''.join(random.choices(string.ascii_lowercase + string.ascii_uppercase, k=length))

def string_from_number(n, length=256):
    d = dict(enumerate('abcdefghjk'))
    # s = "{:0256}".format(n)
    s = ''.join(d[int(c)] for c in str(n))
    s = 'a'*(length-len(s) - (n%128)) + s
    return s
    
n = int(sys.argv[1])
for i in random.sample(range(n), k=n):
    print("+ {} {}".format(string_from_number(i), i))
    
print("! Save dict")
print("! Load dict")
   
for i in range(n//2):
    x = random.randint(0, n)
    print("- {}".format(string_from_number(x), x))
