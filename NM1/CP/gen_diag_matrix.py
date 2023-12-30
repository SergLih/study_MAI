import sys

def main():
    if len(sys.argv) < 3:
        print('usage: python gen_diag_matrix.py <size_matrix> <output>')
        sys.exit(1) 

    size = int(sys.argv[1])
    outfilename = sys.argv[2]

    if size < 3:
        sys.exit()

    with open(outfilename, 'w') as f_out:        
        print(size, file=f_out)
        print('-2 1', file=f_out)
        for _ in range(size - 2):
            print('1 -2 1', file=f_out)
        print('1 -2', file=f_out)
        print('-1 ', file=f_out)
        for _ in range(size - 2):
            print('0 ', file=f_out)
        print('-1', file=f_out)
        
if __name__ == "__main__":
    main()
