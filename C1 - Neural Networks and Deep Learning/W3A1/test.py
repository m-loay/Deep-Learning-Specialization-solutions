import numpy as np

a = np.ones((3, 3))
b = np.ones((3, 1)) * 2
c = a * b
d = np.dot(a, b)
print(a)
print(b)
print(c)
print(d)
