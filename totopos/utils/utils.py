import torch

def generalized_distance_matrix_torch(X,Y):
    """
    Returns the distances between all datapoints from X and Y,
    considering each datapoint to be a row vector in each matrix.

    Notes
    -----
    Let $X \in R^{n_x \times k}, Y \in R^{n_y \times k}$, the distance between all
    points in X to all points in Y is:

    $D = \mathrm{diag}(XX^T) \mathbf{1}^T_{n_y} + \mathbf{1}_{n_x} \mathrm{diag}(YY^T)^T -2 XY^T$

    where $D_{(i,j)} = || x_i - x_j ||^2$ , i.e. the squared euclidean distance.
    """
    n_x,k_x = X.shape
    n_y,k_y = Y.shape
    dev = X.device

    assert k_x == k_y, 'Number of cols of data X is %d and of Y is %d'%(k_x, k_y) # dimensionality of vector spaces must be equal


    diag_x = torch.zeros((n_x, 1)).to(dev)
    diag_y = torch.zeros((1, n_y)).to(dev)

    for i in range(n_x):
        diag_x[i] = torch.dot(X[i], X[i])

    for j in range(n_y):
        diag_y[0, j] = torch.dot(Y[j], Y[j])

    g1 = diag_x @ torch.ones((1, n_y), device=dev)
    g2 = torch.ones((n_x, 1), device=dev) @ diag_y

    D = g1 + g2 - 2*X@Y.T

    return torch.sqrt(D)