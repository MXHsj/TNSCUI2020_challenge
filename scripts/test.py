import numpy as np
import os
import joblib

tag = np.array([[8, 0], [3, 1], [5, 0]])

tag = tag[tag[:, 0].argsort()]

n = 36.2554654
print('{:.2f}'.format(n))

print(tag)
for i in range(5, 10):
    print(i)
