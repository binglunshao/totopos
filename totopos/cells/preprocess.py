import numpy as np
from scipy.sparse import linalg as sla

def find_pca_cutoff(adata, thres=0.02, k=100, save_pca=True):
    """
    Find the optimal number of principal components based on explained variance gaps.
    
    Parameters:
    -----------
    adata : AnnData object
        Annotated data object containing the expression matrix
    thres : float, default=0.02
        Threshold parameter for gap detection (typically between 0 and 1)
    k : int, default=100
        Number of singular values to compute
    
    Returns:
    --------
    cutoff_pc : int
        The recommended number of principal components to retain
    """
    
    # Center the data
    X_cen = adata.X - adata.X.mean(axis=0)
    
    # Perform SVD
    U, s, Vt = sla.svds(X_cen, k=k)
    ix_sort = np.argsort(s)[::-1]
    U, s, Vt = U[:, ix_sort], s[ix_sort], Vt[ix_sort, :]
    explained_variance_ratio = s**2

    # Find gaps between consecutive components
    gaps = explained_variance_ratio[:-1] - explained_variance_ratio[1:]
    
    # Apply threshold
    gap_threshold = thres * explained_variance_ratio[:-1]
    below_thresh = np.where(gaps < gap_threshold)[0]
    
    # Determine cutoff
    cutoff_pc = below_thresh[0] + 1 if len(below_thresh) > 0 else k
    
    if save_pca:
        pcs = U*s
        adata.obsm['pcs'] = pcs
        adata.obs['pc1'] = pcs[:, 0]
        adata.obs['pc2'] = pcs[:, 1]
        adata.obs['pc3'] = pcs[:, 2]
        adata.obs['pc4'] = pcs[:, 3]

    return cutoff_pc

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_outliers(adata, n_pcs=50, contamination=0.05, max_samples=100000, 
                    n_jobs=-1, n_estimators=100, random_state=42):
    """
    Detect outliers using Isolation Forest on PCA data.
    
    Parameters:
    -----------
    adata : AnnData
        Data with PCA in adata.obsm['X_pca'] or adata.obsm['pca']
    n_pcs : int, default=50
        Number of PCs to use
    contamination : float, default=0.05
        Expected fraction of outliers
    max_samples : int, default=100000
        Subsample size for tree building
    
    Returns:
    --------
    outlier_mask : array
        Boolean mask (True = outlier)
    outlier_scores : array
        Anomaly scores (more negative = more anomalous)
    """
    
    # Get PCA data
    if 'X_pca' in adata.obsm:
        pca_data = adata.obsm['X_pca']
    elif 'pca' in adata.obsm:
        pca_data = adata.obsm['pca']
    elif 'pcs' in adata.obsm:
        pca_data = adata.obsm['pcs']
    else:
        raise ValueError("No PCA found. Run sc.pp.pca() first.")
    
    # Use specified number of PCs
    features = pca_data[:, :n_pcs]
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Fit Isolation Forest
    clf = IsolationForest(
        n_estimators=n_estimators,
        max_samples=min(max_samples, len(features)),
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    predictions = clf.fit_predict(features_scaled)
    scores = clf.score_samples(features_scaled)
    
    outlier_mask = predictions == -1
    
    print(f"Found {np.sum(outlier_mask):,} outliers ({100*np.sum(outlier_mask)/len(outlier_mask):.1f}%)")
    
    return outlier_mask, scores

def plot_outliers(adata, outlier_mask, outlier_scores):
    """
    Simple visualization of outlier detection results.
    """
    
    # Get PCA coordinates
    if 'X_pca' in adata.obsm:
        pca = adata.obsm['X_pca']
    elif 'pca' in adata.obsm:
        pca = adata.obsm['pca']
    elif 'pcs' in adata.obsm:
        pca = adata.obsm['pcs']
    else:
        raise ValueError("No PCA found. Run sc.pp.pca() first.")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: PCA with outliers highlighted
    normal = ~outlier_mask
    axes[0].scatter(pca[normal, 0], pca[normal, 1], c='lightblue', s=1, alpha=0.6, label='Normal')
    axes[0].scatter(pca[outlier_mask, 0], pca[outlier_mask, 1], c='red', s=3, alpha=0.8, label='Outliers')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()
    axes[0].set_title('Outliers in PC Space')
    
    # Plot 2: Score distribution
    axes[1].hist(outlier_scores[normal], bins=50, alpha=0.7, label='Normal', color='lightblue')
    axes[1].hist(outlier_scores[outlier_mask], bins=20, alpha=0.8, label='Outliers', color='red')
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].set_title('Score Distribution')
    
    # Plot 3: QC comparison (if available)
    if 'total_counts' in adata.obs.columns:
        axes[2].boxplot([adata.obs.loc[normal, 'total_counts'], 
                        adata.obs.loc[outlier_mask, 'total_counts']], 
                       labels=['Normal', 'Outliers'])
        axes[2].set_ylabel('Total Counts')
        axes[2].set_yscale('log')
        axes[2].set_title('Total Counts')
    else:
        axes[2].text(0.5, 0.5, 'No QC metrics\navailable', ha='center', va='center')
        axes[2].set_title('QC Metrics')
    
    plt.tight_layout()
    plt.show()

# Simple usage example:
"""
# Detect outliers
outlier_mask, scores = detect_outliers(adata, n_pcs=50, contamination=0.05)

# Visualize results
plot_outliers(adata, outlier_mask, scores)

# Add to adata and filter
adata.obs['is_outlier'] = outlier_mask
adata.obs['outlier_score'] = scores
adata_clean = adata[~outlier_mask].copy()
"""