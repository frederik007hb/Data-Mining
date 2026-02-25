import numpy as np
def encode_results(results):
    """
    Endcoding of results
    1: HOME WIN, -1: AWAY WIN, 0: DRAW

    Args:
    results: list of dicts - results to encode 
    """
    encoding = np.zeros(len(results))

    for i in range(len(results)):
        if results[i] == "H":
            encoding[i] = 1
        elif results[i] == "A":
            encoding[i] = -1
        else:
            encoding[i] = 0
    return encoding

def one_hot_encode_strings(vec):
    """
    One-hot encode an array of categorical strings.
    
    Args:
        vec: numpy array of strings
        
    Returns:
        enc: one-hot encoded matrix
        class_to_index: dictionary mapping class name -> index
    """
    classes = np.unique(vec)
    k = len(classes)
    
    # Create mapping
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    
    # Convert strings to integer labels
    int_labels = np.array([class_to_index[x] for x in vec])
    
    # One-hot encoding
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), int_labels] = 1
    
    return enc, class_to_index

def clique_step1_dense_cells(X, xi=3, tau=0.21, domain=(0, 100)):
    """
    Step 1 of CLIQUE: Detect dense 1D cells.
    
    Parameters:
        X      : numpy array (n_samples, n_features)
        xi     : number of equal intervals per dimension
        tau    : density threshold (fraction, e.g., 0.21)
        domain : tuple (min, max) defining global domain
        
    Returns:
        dense_cells : dict {dimension: list of dense interval indices}
    """
    
    n, d = X.shape
    min_points = int(np.ceil(tau * n))
    
    # Create equal-width intervals
    interval_edges = np.linspace(domain[0], domain[1], xi + 1)
    
    dense_cells = {}
    
    for dim in range(d):
        dense_cells[dim] = []
        
        for i in range(xi):
            low = interval_edges[i]
            high = interval_edges[i + 1]
            
            # Include upper bound in last interval
            if i == xi - 1:
                mask = (X[:, dim] >= low) & (X[:, dim] <= high)
            else:
                mask = (X[:, dim] >= low) & (X[:, dim] < high)
            
            count = np.sum(mask)
            
            if count >= min_points:
                dense_cells[dim].append(i)   # store interval index
        
    return dense_cells

def euc_distance(x, y):
    """
    Compute Euclidean distance between two points.
    
    Parameters:
        x : numpy array (n_features,)
        y : numpy array (n_features,)
        
    Returns:
        dist : float
    """
    return np.sqrt(np.sum((x - y) ** 2))

def contingency_table(C, T, ignore_noise=True): 

    """
    Args:
        C(numpy array):       Clusters obtained by a clustering algorithm as a nx1 vector
        T(numpy array):       Ground-truth cluster assignments as a nx1 vector

    Returns:
        ctable:   a num_clusters_of_C x num_cluster_of_T matrix containing the overlaps among the different clusters
    """
    
    if ignore_noise:
        mask = (C != -1)
        C_masked = C[mask]
        indices_masked = np.where(mask)[0]
    else:
        C_masked = C
        indices_masked = np.arange(len(C))

    clusters = np.unique(C_masked)
    classes  = np.unique(T)   # include all classes, not just masked
    
    ctable = np.zeros((len(clusters), len(classes)), dtype=int)

    for i, c in enumerate(clusters):
        for j, t in enumerate(classes):
            # count overlap only for points that are not noise
            count = np.sum((C_masked == c) & (T[indices_masked] == t))
            ctable[i, j] = count

    return ctable

def purity_score(ctable):
    """
    Args:
        ctable (numpy array):  Contingency table of shape (num_clusters, num_classes) containing the overlaps among the different clusters
    
    Returns:
        purity(float):   a float, the purity of clustering result
    """
    
    clusters, gt_classes = ctable.shape
    purity = np.zeros(clusters)

    for i in range(clusters):
        purity[i] = np.max(ctable[i])/np.sum(ctable[i]) if np.sum(ctable[i]) > 0 else 0
    return purity