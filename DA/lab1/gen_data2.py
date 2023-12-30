#!/usr/bin/python

infile  = input('Input file: ')
outfile = input('Output file: ')

with open(infile, 'r') as f_in:
    with open(outfile, 'w') as f_out:
        lines = f_in.readlines()
        f_out.writelines(lines[::-1])
