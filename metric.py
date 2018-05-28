import numpy as np

def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes
    
    :param clusters: the cluster assignments array
    :type clusters: numpy.array
    
    :param classes: the ground truth classes
    :type classes: numpy.array
    
    :returns: the purity score
    :rtype: float
    """
    
    A = np.c_[(clusters,classes)]

    n_accurate = 0.

    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]

def f1_score(clusters, classes):
    N = len(clusters)
    relation = np.zeros((N, N))
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    for i in range(N):
        for j in range(i+1):
            if clusters[i] == clusters[j] and classes[i] == classes[j]:
                TP += 1.0
            elif clusters[i] == clusters[j] and classes[i] != classes[j]:
                FP += 1.0
            elif clusters[i] != clusters[j] and classes[i] == classes[j]:
                FN += 1.0
            else:
                TN += 1.0

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    f1 = (2.0 * P * R) / (P + R)

    return P, R, f1
