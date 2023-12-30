import numpy as np
import sys
import matplotlib.pyplot as plt

interval = (1, 2)

if len(sys.argv) < 4:
    print('usage: python 2_1.py <eps> <output.txt> <log.txt>')
    sys.exit(1)
    
EPS = sys.argv[1]
outfilename = sys.argv[2]
logfilename = sys.argv[3]

def f(x):
    return x * x * x + x * x - 2 * x - 1

def df(x):
    return 3 * x * x + 2 * x - 2
    
#def phi(x):
#    return (x**3 + x**2 - 1) / 2

#def dphi(x):
#    return (3*x**2 + 2*x) / 2

def select_lambda():
    x = np.linspace(interval[0], interval[1], 100)
    y = [df(i) for i in x]
    sign_df = None
    if all([np.sign(i) == -1 for i in y]):
        sign_df = -1
    elif all([np.sign(i) == 1 for i in y]):
        sign_df = 1
    else:
        with open(logfilename, 'at') as f:
            print('Error: Derivative change sign\n', file=f)
            sys.exit(1)
    y = [abs(df(i)) for i in x]
    return sign_df / max(y)

def phi(x):
    lmbd = select_lambda()
    return x - lmbd * f(x)

def dphi(x):
    lmbd = select_lambda()
    return 1 - lmbd * df(x)   
   
with open(logfilename, 'wt') as f_log:
    print('Function: x^3 + x^2 - 2 * x - 1 ', file=f_log)
    print('\nDerivative function: 3 * x^2 + 2 * x - 2 ', file=f_log)
    print('\nPrecision: ', EPS, file=f_log)
    print('\nInterval: ', interval, file=f_log)
    print(file=f_log)
    
def get_q():
    return max(abs(dphi(interval[0])), abs(dphi(interval[1])))
   
def newton_method():
    x = interval[1]
    cnt_iter = 0
    #points = [x]
    with open(logfilename, 'at') as f_log:
        print('Newton method:\n ', file=f_log)
    while True:
        x_next = x - f(x)/df(x)
        with open(logfilename, 'at') as f_log:
            print('Iter: ', cnt_iter, '\t x_i: ', round(x, 5), '\t f(x_i): ', round(f(x), 5), '\t df(x_i): ', round(df(x), 5), '\t-f(x)/df(x): ', -f(x)/df(x), file=f_log)
            print(file=f_log)
        cnt_iter += 1
        if abs(x_next - x) < float(EPS):
            return x_next#, points
        else:
            #points.append(x_next)
            x = x_next

def draw_graphic():
    x = np.linspace(0, 2, 1000)
    def f1(x):
        return x*x*x
    y = [f1(i) for i in x]
    plt.plot(x, y)

    def f2(x):
        return 2*x - x*x + 1
    y = [f2(i) for i in x]

    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    #for i in points:
    #    plt.plot(i, f1(i), 'or')
    #    plt.plot(i, f2(i), 'or')    
    plt.grid(True)
    plt.legend(['y = x^3', 'y = 2x - x^2 + 1'], loc='upper left')
    plt.grid(True)
    plt.savefig('Interval_definition')
    #plt.show()
    
draw_graphic()
    
def iteration_method():
    x = (interval[1] + interval[0]) /  2
    cnt_iter = 0
    q = get_q()
    with open(logfilename, 'at') as f_log:
        print('Iteration method:\nSelect q:  ', q, '\n', file=f_log)
    #points = [x]
    while True:
        x_next = phi(x)
        with open(logfilename, 'at') as f_log:
            print('Iter: ', cnt_iter, '\tx_i: ', x, '\tPhi(x_i): ', phi(x), file=f_log)
            print(file=f_log)
        cnt_iter += 1
        if q * abs(x_next - x) / (1 - q) < float(EPS):
            return x_next#, points
        #points.append(x_next)
        x = x_next

with open(outfilename, 'wt') as f_out:
    print("Newton method: ", newton_method(), file=f_out)
    print("\nIteration_method: ", iteration_method(), file=f_out)