from chvector.models.archetypes import *
from chvector.models.solver import *

import numpy as np

N = 3

f = sinusoid(1, 1.2, 0.7, N)
f = f[np.newaxis, np.newaxis, :]

# Needs correct weights!!!!!
U = sinusoid_pair(N)

p, delt, lamb = create_poly(f, U)

p = p[:,:,::2]

r = poly_roots(p)

v = poly_value(p, np.angle(r))

idx = np.argmax(v, axis=2)
th2 = np.angle(r)
th2 = np.take_along_axis(th2, idx[:,:,np.newaxis], axis=2)
th2 /= 2

s0 = poly_value(lamb[0], th2)
s1 = poly_value(lamb[1], th2)

Ar = np.sqrt(s0**2 + s1**2)
phi2 = np.arctan2(s1, s0)

print(Ar)
print(phi2)
print(th2)