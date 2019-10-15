import numpy as np
import numba as nb
from math import atan2, floor, pi, ceil, isnan

@nb.jit(nopython=True, parallel=True)
def hashkey(block, Qangle, W, nbgx, nbgy):
    # Calculate gradient
    gy = nbgy
    gx = nbgx

    # Transform 2D matrix into 1D array
    gx = gx.ravel()
    gy = gy.ravel()

    # SVD calculation
    G = np.vstack((gx,gy)).T
    
    x0 = np.dot(G.T,W)
    x = np.dot(x0, G)
    w, v = np.linalg.eig(x);

    # Make sure V and D contain only real numbers
    # nonzerow = np.count_nonzero(np.isreal(w))
    # nonzerov = np.count_nonzero(np.isreal(v))
    # if nonzerow != 0:
    # w = np.real(w)
    # if nonzerov != 0:
    # v = np.real(v)

    # Sort w and v according to the descending order of w
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:,idx]

    # Calculate theta
    theta = atan2(v[1,0], v[0,0])
    if theta < 0:
        theta = theta + pi

    # # Calculate lamda A
    # lamda = w[0]

    # # Calculate u
    # sqrtlamda1 = np.sqrt(w[0])
    # sqrtlamda2 = np.sqrt(w[1])
    # if sqrtlamda1 + sqrtlamda2 == 0:
    #     u = 0
    # else:
    #     u = (sqrtlamda1 - sqrtlamda2)/(sqrtlamda1 + sqrtlamda2)

    # PART B BELOW

    theta = floor(theta/(pi/Qangle))
    lamda = w[0]
    u = (np.sqrt(w[0]) - np.sqrt(w[1]))/(np.sqrt(w[0]) + np.sqrt(w[1]) + 0.00000000000000001)
    if isnan(u):
        u=1
    #================================== B ends
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
