import os
import sys

processor = sys.argv[1]
arguments = sys.argv[1:]
count = len(arguments)

if (count > 2):
    print("python3 <name_program> (--cpu or --gpu) (--default)")
    exit(1)

if (sys.argv[-1] == '--default'):
    os.system('cat /home/sergey/MAI/PGP/kp_tracer/test_for_kp_pgp')

if (processor == '--cpu'):
    os.system('/home/sergey/MAI/PGP/kp_tracer/kp/cpu')
elif (count == 0 or (processor == '--gpu')):
    os.system('/home/sergey/MAI/PGP/kp_tracer/kp_cuda/gpu')

    

    
