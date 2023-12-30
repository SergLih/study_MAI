import math
from md5_old import md5_old

import hashlib


#dic = dict()
#a =list(hashlib.algorithms_guaranteed)

def md5_built_in(s):
    return hashlib.md5(s).hexdigest()

rotate_amounts = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                  5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
                  4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                  6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21]

tsin = [int(abs(math.sin(i + 1)) * 2 ** 32) & 0xFFFFFFFF for i in range(64)]

init_values = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476]

functions = [lambda b, c, d: (b & c) | (~b & d), 
             lambda b, c, d: (d & b) | (~d & c),
             lambda b, c, d: b ^ c ^ d ,
             lambda b, c, d: c ^ (b | ~d)]

index_functions = [lambda i: i,
                   lambda i: (5 * i + 1) % 16,
                   lambda i: (3 * i + 5) % 16,
                   lambda i: (7 * i) % 16]


def left_rotate(x, amount):
    x &= 0xFFFFFFFF
    return ((x << amount) | (x >> (32 - amount))) & 0xFFFFFFFF



def md5(message_, rounds=4):
    message = bytearray(message_)  # copy our input into a mutable buffer
    orig_len_in_bits = (8 * len(message)) & 0xffffffffffffffff
    message.append(0x80)
    while len(message) % 64 != 56:
        message.append(0)
    message += orig_len_in_bits.to_bytes(8, byteorder='little')

    hash_pieces = init_values[:]

    for chunk_ofst in range(0, len(message), 64):   # блоки по 64 байта (512 бит)
        a, b, c, d = hash_pieces
        chunk = message[chunk_ofst:chunk_ofst + 64]  # берем очередной блок
        for rr in range(rounds):
            r = rr % 4
            for kk in range(16):
                i = r*16 + kk
                k = index_functions[r](i)   # вычисляем номер под-блока данных (0...15)
                x_k = int.from_bytes(chunk[4*k : 4*k+4], byteorder='little')
                f = functions[r](b, c, d)
                to_rotate = a + f + tsin[i] + x_k
                new_b = (b + left_rotate(to_rotate, rotate_amounts[i])) & 0xFFFFFFFF
                a, b, c, d = d, new_b, b, c
        for i, val in enumerate([a, b, c, d]):
            hash_pieces[i] += val
            hash_pieces[i] &= 0xFFFFFFFF
        print(hash_pieces)

    return sum(x << (32 * i) for i, x in enumerate(hash_pieces))


def md5_to_hex(digest):
    raw = digest.to_bytes(16, byteorder='little')
    return '{:032x}'.format(int.from_bytes(raw, byteorder='big'))


# if __name__ == '__main__':
demo = [
    # "", "a", "abc", "message digest", "abcdefghijklmnopqrstuvwxyz",
    #     "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "12345asdfg"]
        # "12345678901234567890123456789012345678901234567890123456789012345678901234567890"]
for message in demo:
    print(md5_to_hex(md5(message.encode('utf-8'), rounds=8)), ' <= "', message, '"', sep='')
    print('--------------------')
    print(md5_to_hex(md5_old(message.encode('utf-8'))))
    print(md5_built_in(message.encode('utf-8')))
    print()