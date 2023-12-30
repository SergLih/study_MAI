import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

delta = 0.025
l1 = 4
l2 = 5
X = np.arange(0, l1+delta, delta)
Y = np.arange(0, l2+delta, delta)
x, y = np.meshgrid(X, Y)

z = np.sin((np.pi / l1) * x) * (-np.exp((-np.pi / l1) * y) + np.exp((np.pi / l1) * y)) / ((np.pi / l1) * (np.exp((np.pi / l1) * l2) + np.exp((-np.pi / l1) * l2)))

fig, ax = plt.subplots(figsize=(l1, l2))
CS = ax.contour(x, y, z)
for i, label in enumerate(CS.cvalues): 
    CS.collections[i].set_label(np.round(label, 1))
ax.legend(loc='lower right')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()