import numpy as np
import numba as nb
from math import atan2, floor, pi


def hashkey(Qangle, W, gy, gx):
	x = hashkey1(W, gy, gx)
	cx = np.array(x, dtype = np.complex128)
	return hashkey2(Qangle, cx)


@nb.jit(nopython=True, parallel=True)
def hashkey1(W, gy, gx):
	# Transform 2D matrix into 1D array
	gx = gx.ravel()
	gy = gy.ravel()
	# SVD calculation
	G = np.vstack((gx,gy)).T    
	x0 = np.dot(G.T,W)
	x = np.dot(x0, G)
	return x

@nb.jit(nopython=True, parallel=True)
def hashkey2(Qangle, x):

	w, v = np.linalg.eig(x)
	w = w.real
	v = v.real

	idx = w.argsort()[::-1]
	w = w[idx]
	v = v[:,idx]

	# Calculate theta
	theta = atan2(v[1,0], v[0,0])
	if theta < 0:
		theta = theta + pi

	# Calculate lamda
	lamda = w[0]

	# Calculate u
	sqrtlamda1 = np.sqrt(w[0])
	sqrtlamda2 = np.sqrt(w[1])
	if sqrtlamda1 + sqrtlamda2 == 0:
		u = 0
	else:
		u = (sqrtlamda1 - sqrtlamda2)/(sqrtlamda1 + sqrtlamda2)

	# Quantize
	angle = floor(theta/pi*Qangle)
	if lamda < 0.0001:
		strength = 0
	elif lamda > 0.001:
		strength = 2
	else:
		strength = 1
	if u < 0.25:
		coherence = 0
	elif u > 0.5:
		coherence = 2
	else:
		coherence = 1

	# Bound the output to the desired ranges
	if angle > 23:
		angle = 23
	elif angle < 0:
		angle = 0

	return angle, strength, coherence
