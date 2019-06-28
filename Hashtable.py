import numpy as ny
def hashTable(patch,Qangle,Qstrenth,Qcoherence):
    #convert the matrix into 2-D array(convert the  value of matrix into patches and stores as array) ...
    [gx,gy] = ny.gradient(patch)
    G = ny.matrix((gx.ravel(),gy.ravel())).T
    x = G.T*G
    [eigenvalues,eigenvectors] = ny.linalg.eig(x)
    #For angle
    angle = ny.math.atan2(eigenvectors[0,1],eigenvectors[0,0])
    if angle<0:
        angle += ny.pi
    #For strength
    strength = eigenvalues.max()/(eigenvalues.sum()+0.0001)
    #For coherence
    lamda1 = ny.math.sqrt(eigenvalues.max())
    lamda2 = ny.math.sqrt(eigenvalues.min())
    coherence = ny.abs((lamda1-lamda2)/(lamda1+lamda2+0.0001))
    #Quantization
    angle = ny.floor(angle/(np.pi/Qangle)-1)
    strength = ny.floor(strength/(1.0/Qstrenth)-1)
    coherence = ny.floor(coherence/(1.0/Qcoherence)-1)
    return int(angle),int(strength),int(coherence)
