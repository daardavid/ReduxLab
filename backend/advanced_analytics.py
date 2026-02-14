"""
Advanced Analytics Module for ReduxLab

Provides additional statistical tests, dimensionality reduction methods,
clustering algorithms, data transformations, and network analysis tools.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Statistical Tests (Pre-PCA suitability)
# ============================================================================

def bartlett_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Bartlett's test of sphericity.

    Tests whether the correlation matrix is an identity matrix,
    i.e. whether variables are unrelated and therefore unsuitable for PCA.

    Args:
        df: Numeric DataFrame (samples x features).

    Returns:
        Dict with 'chi_square', 'p_value', and 'suitable_for_pca'.
    """
    from scipy import stats

    n, p = df.shape
    corr = df.corr()
    det = np.linalg.det(corr.values)

    if det <= 0:
        det = 1e-18

    chi_square = -((n - 1) - (2 * p + 5) / 6) * np.log(det)
    dof = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi_square, dof)

    result = {
        'chi_square': float(chi_square),
        'degrees_of_freedom': int(dof),
        'p_value': float(p_value),
        'suitable_for_pca': p_value < 0.05,
        'interpretation': (
            "Variables are correlated; PCA is appropriate."
            if p_value < 0.05
            else "Variables may be uncorrelated; PCA may not be appropriate."
        )
    }
    logger.info(f"Bartlett's test: chi2={chi_square:.2f}, p={p_value:.4f}, suitable={result['suitable_for_pca']}")
    return result


def kmo_test(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy.

    Measures the proportion of variance among variables that might be
    common variance. Higher values (> 0.6) indicate PCA is suitable.

    Args:
        df: Numeric DataFrame (samples x features).

    Returns:
        Dict with 'overall_kmo', per-variable 'kmo_per_variable', and 'interpretation'.
    """
    corr = df.corr().values
    n = corr.shape[0]

    # Partial correlation matrix
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr)

    # Diagonal scaling for partial correlations
    diag = np.diag(1.0 / np.sqrt(np.diag(inv_corr)))
    partial = -diag @ inv_corr @ diag
    np.fill_diagonal(partial, 0)

    # KMO per variable and overall
    corr_sq = corr ** 2
    partial_sq = partial ** 2
    np.fill_diagonal(corr_sq, 0)

    sum_corr_sq = corr_sq.sum(axis=0)
    sum_partial_sq = partial_sq.sum(axis=0)

    kmo_per_var = sum_corr_sq / (sum_corr_sq + sum_partial_sq)
    kmo_overall = corr_sq.sum() / (corr_sq.sum() + partial_sq.sum())

    # Interpretation
    if kmo_overall >= 0.9:
        quality = "Marvelous"
    elif kmo_overall >= 0.8:
        quality = "Meritorious"
    elif kmo_overall >= 0.7:
        quality = "Middling"
    elif kmo_overall >= 0.6:
        quality = "Mediocre"
    elif kmo_overall >= 0.5:
        quality = "Miserable"
    else:
        quality = "Unacceptable"

    result = {
        'overall_kmo': float(kmo_overall),
        'kmo_per_variable': dict(zip(df.columns, kmo_per_var.tolist())),
        'quality': quality,
        'suitable_for_pca': kmo_overall >= 0.6,
        'interpretation': f"KMO = {kmo_overall:.3f} ({quality}). "
                          f"{'PCA is appropriate.' if kmo_overall >= 0.6 else 'Consider removing weakly correlated variables.'}"
    }
    logger.info(f"KMO test: overall={kmo_overall:.3f} ({quality})")
    return result


def cronbach_alpha(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Cronbach's alpha for internal consistency / reliability.

    Args:
        df: Numeric DataFrame (samples x items/features).

    Returns:
        Dict with 'alpha', 'interpretation', and 'item_total_correlations'.
    """
    n_items = df.shape[1]
    item_variances = df.var(ddof=1)
    total_variance = df.sum(axis=1).var(ddof=1)

    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)

    # Item-total correlations
    total = df.sum(axis=1)
    item_total = {}
    for col in df.columns:
        rest = total - df[col]
        item_total[col] = float(df[col].corr(rest))

    if alpha >= 0.9:
        quality = "Excellent"
    elif alpha >= 0.8:
        quality = "Good"
    elif alpha >= 0.7:
        quality = "Acceptable"
    elif alpha >= 0.6:
        quality = "Questionable"
    elif alpha >= 0.5:
        quality = "Poor"
    else:
        quality = "Unacceptable"

    return {
        'alpha': float(alpha),
        'quality': quality,
        'n_items': n_items,
        'item_total_correlations': item_total,
        'interpretation': f"Cronbach's alpha = {alpha:.3f} ({quality})"
    }


def mahalanobis_distance(df: pd.DataFrame) -> pd.Series:
    """
    Compute Mahalanobis distance for each observation (multivariate outlier detection).

    Args:
        df: Numeric DataFrame (samples x features).

    Returns:
        pd.Series of Mahalanobis distances indexed like df.
    """
    mean = df.mean().values
    cov = df.cov().values

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    diff = df.values - mean
    left = diff @ inv_cov
    distances = np.sqrt(np.sum(left * diff, axis=1))

    return pd.Series(distances, index=df.index, name='mahalanobis_distance')


# ============================================================================
# Dimensionality Reduction
# ============================================================================

def factor_analysis(df: pd.DataFrame, n_factors: int = 2,
                    rotation: str = 'varimax') -> Dict[str, Any]:
    """
    Factor Analysis with optional rotation.

    Args:
        df: Standardized numeric DataFrame.
        n_factors: Number of latent factors.
        rotation: Rotation method ('varimax', 'promax', or None).

    Returns:
        Dict with 'loadings', 'communalities', 'explained_variance', 'scores'.
    """
    from sklearn.decomposition import FactorAnalysis

    fa = FactorAnalysis(n_components=n_factors, rotation=rotation)
    scores = fa.fit_transform(df)
    loadings = pd.DataFrame(
        fa.components_.T,
        index=df.columns,
        columns=[f"Factor_{i+1}" for i in range(n_factors)]
    )
    communalities = pd.Series(
        1 - fa.noise_variance_,
        index=df.columns,
        name='communality'
    )
    explained = fa.noise_variance_

    logger.info(f"Factor Analysis: {n_factors} factors, rotation={rotation}")
    return {
        'loadings': loadings,
        'communalities': communalities,
        'noise_variance': pd.Series(explained, index=df.columns),
        'scores': pd.DataFrame(scores, index=df.index,
                               columns=[f"Factor_{i+1}" for i in range(n_factors)]),
        'model': fa
    }


def run_tsne(df: pd.DataFrame, n_components: int = 2,
             perplexity: float = 30.0, random_state: int = 42) -> Dict[str, Any]:
    """
    t-SNE dimensionality reduction.

    Args:
        df: Numeric DataFrame.
        n_components: 2 or 3 for visualization.
        perplexity: t-SNE perplexity parameter.
        random_state: Random seed for reproducibility.

    Returns:
        Dict with 'embedding' DataFrame and 'params'.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, n_iter=1000)
    embedding = tsne.fit_transform(df.values)

    cols = [f"t-SNE_{i+1}" for i in range(n_components)]
    df_result = pd.DataFrame(embedding, index=df.index, columns=cols)

    logger.info(f"t-SNE: {n_components}D, perplexity={perplexity}, KL_divergence={tsne.kl_divergence_:.4f}")
    return {
        'embedding': df_result,
        'kl_divergence': float(tsne.kl_divergence_),
        'params': {'n_components': n_components, 'perplexity': perplexity}
    }


def run_umap(df: pd.DataFrame, n_components: int = 2,
             n_neighbors: int = 15, min_dist: float = 0.1,
             random_state: int = 42) -> Dict[str, Any]:
    """
    UMAP dimensionality reduction.

    Requires: pip install umap-learn

    Args:
        df: Numeric DataFrame.
        n_components: 2 or 3 for visualization.
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance parameter.
        random_state: Random seed.

    Returns:
        Dict with 'embedding' DataFrame and 'params'.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "UMAP requires the umap-learn package. Install with: pip install umap-learn"
        )

    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state)
    embedding = reducer.fit_transform(df.values)

    cols = [f"UMAP_{i+1}" for i in range(n_components)]
    df_result = pd.DataFrame(embedding, index=df.index, columns=cols)

    logger.info(f"UMAP: {n_components}D, n_neighbors={n_neighbors}, min_dist={min_dist}")
    return {
        'embedding': df_result,
        'params': {'n_components': n_components, 'n_neighbors': n_neighbors, 'min_dist': min_dist}
    }


def run_ica(df: pd.DataFrame, n_components: int = 2,
            random_state: int = 42) -> Dict[str, Any]:
    """
    Independent Component Analysis (ICA).

    Args:
        df: Numeric DataFrame.
        n_components: Number of independent components.
        random_state: Random seed.

    Returns:
        Dict with 'components', 'mixing_matrix', 'sources'.
    """
    from sklearn.decomposition import FastICA

    ica = FastICA(n_components=n_components, random_state=random_state)
    sources = ica.fit_transform(df.values)

    cols = [f"IC_{i+1}" for i in range(n_components)]
    df_sources = pd.DataFrame(sources, index=df.index, columns=cols)
    df_mixing = pd.DataFrame(
        ica.mixing_,
        index=df.columns,
        columns=cols
    )

    logger.info(f"ICA: {n_components} components, iterations={ica.n_iter_}")
    return {
        'sources': df_sources,
        'mixing_matrix': df_mixing,
        'n_iter': ica.n_iter_,
        'model': ica
    }


# ============================================================================
# Clustering
# ============================================================================

def kmeans_analysis(df: pd.DataFrame, max_k: int = 10,
                    random_state: int = 42) -> Dict[str, Any]:
    """
    K-Means with elbow method and silhouette analysis.

    Args:
        df: Numeric DataFrame.
        max_k: Maximum number of clusters to test.
        random_state: Random seed.

    Returns:
        Dict with 'inertias', 'silhouette_scores', 'optimal_k', 'labels', 'model'.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    max_k = min(max_k, len(df) - 1)
    inertias = []
    silhouettes = []

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(df)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(df, labels)))

    # Optimal k = highest silhouette
    optimal_idx = int(np.argmax(silhouettes))
    optimal_k = optimal_idx + 2

    # Fit final model
    final_km = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=10)
    final_labels = final_km.fit_predict(df)

    logger.info(f"K-Means: optimal_k={optimal_k}, best_silhouette={silhouettes[optimal_idx]:.3f}")
    return {
        'k_range': list(range(2, max_k + 1)),
        'inertias': inertias,
        'silhouette_scores': silhouettes,
        'optimal_k': optimal_k,
        'labels': pd.Series(final_labels, index=df.index, name='cluster'),
        'centers': pd.DataFrame(final_km.cluster_centers_, columns=df.columns),
        'model': final_km
    }


def dbscan_analysis(df: pd.DataFrame, eps: float = 0.5,
                    min_samples: int = 5) -> Dict[str, Any]:
    """
    DBSCAN density-based clustering.

    Args:
        df: Numeric DataFrame.
        eps: Maximum distance between two samples in the same neighborhood.
        min_samples: Minimum samples in a neighborhood to form a core point.

    Returns:
        Dict with 'labels', 'n_clusters', 'n_noise', 'core_sample_indices'.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(df)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    sil = None
    if n_clusters >= 2 and n_noise < len(labels):
        mask = labels != -1
        if mask.sum() > n_clusters:
            sil = float(silhouette_score(df[mask], labels[mask]))

    logger.info(f"DBSCAN: eps={eps}, min_samples={min_samples}, clusters={n_clusters}, noise={n_noise}")
    return {
        'labels': pd.Series(labels, index=df.index, name='cluster'),
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'core_sample_indices': db.core_sample_indices_.tolist(),
        'silhouette': sil,
        'params': {'eps': eps, 'min_samples': min_samples}
    }


def gmm_analysis(df: pd.DataFrame, max_components: int = 10,
                 random_state: int = 42) -> Dict[str, Any]:
    """
    Gaussian Mixture Model clustering with BIC model selection.

    Args:
        df: Numeric DataFrame.
        max_components: Maximum number of mixture components to test.
        random_state: Random seed.

    Returns:
        Dict with 'labels', 'probabilities', 'bic_scores', 'optimal_n', 'model'.
    """
    from sklearn.mixture import GaussianMixture

    max_components = min(max_components, len(df) - 1)
    bics = []

    for n in range(2, max_components + 1):
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(df)
        bics.append(float(gm.bic(df)))

    optimal_idx = int(np.argmin(bics))
    optimal_n = optimal_idx + 2

    final_gm = GaussianMixture(n_components=optimal_n, random_state=random_state)
    final_gm.fit(df)
    labels = final_gm.predict(df)
    probabilities = final_gm.predict_proba(df)

    logger.info(f"GMM: optimal_n={optimal_n}, best_BIC={bics[optimal_idx]:.2f}")
    return {
        'n_range': list(range(2, max_components + 1)),
        'bic_scores': bics,
        'optimal_n': optimal_n,
        'labels': pd.Series(labels, index=df.index, name='cluster'),
        'probabilities': pd.DataFrame(probabilities, index=df.index),
        'means': pd.DataFrame(final_gm.means_, columns=df.columns),
        'model': final_gm
    }


# ============================================================================
# Data Transformations
# ============================================================================

def yeo_johnson_transform(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Yeo-Johnson power transformation (works with negative values).

    Args:
        df: Numeric DataFrame.

    Returns:
        Tuple of (transformed DataFrame, dict of lambdas per column).
    """
    from scipy.stats import yeojohnson

    transformed = pd.DataFrame(index=df.index)
    lambdas = {}

    for col in df.columns:
        clean = df[col].dropna()
        if len(clean) > 1:
            t_values, lmbda = yeojohnson(clean.values)
            transformed[col] = df[col].copy()
            transformed.loc[clean.index, col] = t_values
            lambdas[col] = float(lmbda)
        else:
            transformed[col] = df[col]
            lambdas[col] = None

    logger.info(f"Yeo-Johnson transformation applied to {len(lambdas)} columns")
    return transformed, lambdas


def winsorize(df: pd.DataFrame, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.DataFrame:
    """
    Winsorize data (cap outliers at specified percentiles).

    Args:
        df: Numeric DataFrame.
        limits: Tuple of (lower_fraction, upper_fraction) to clip.

    Returns:
        Winsorized DataFrame.
    """
    from scipy.stats.mstats import winsorize as scipy_winsorize

    result = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        clean = df[col].dropna()
        if len(clean) > 0:
            winsorized = scipy_winsorize(clean.values, limits=limits)
            result.loc[clean.index, col] = winsorized

    logger.info(f"Winsorization applied: limits={limits}")
    return result


def rank_inverse_normal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank-based inverse normal transformation.

    Transforms each column to approximate normality via rank -> quantile mapping.

    Args:
        df: Numeric DataFrame.

    Returns:
        Transformed DataFrame.
    """
    from scipy.stats import norm

    result = pd.DataFrame(index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col]
        ranks = series.rank(method='average')
        n = series.notna().sum()
        # Blom's formula
        quantiles = (ranks - 0.375) / (n + 0.25)
        result[col] = norm.ppf(quantiles)

    logger.info(f"Rank inverse normal transformation applied to {len(result.columns)} columns")
    return result


def robust_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust scaling using median and IQR (instead of mean and std).

    More robust to outliers than StandardScaler.

    Args:
        df: Numeric DataFrame.

    Returns:
        Robustly scaled DataFrame.
    """
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    scaled = scaler.fit_transform(df)
    result = pd.DataFrame(scaled, index=df.index, columns=df.columns)

    logger.info(f"Robust scaling applied to {len(df.columns)} columns")
    return result


# ============================================================================
# Network Analysis
# ============================================================================

def centrality_measures(graph) -> Dict[str, pd.Series]:
    """
    Compute centrality measures for a NetworkX graph.

    Args:
        graph: NetworkX graph object.

    Returns:
        Dict mapping centrality type to pd.Series of scores.
    """
    import networkx as nx

    results = {}

    results['degree'] = pd.Series(
        dict(nx.degree_centrality(graph)),
        name='degree_centrality'
    )
    results['betweenness'] = pd.Series(
        dict(nx.betweenness_centrality(graph)),
        name='betweenness_centrality'
    )
    results['closeness'] = pd.Series(
        dict(nx.closeness_centrality(graph)),
        name='closeness_centrality'
    )

    try:
        results['eigenvector'] = pd.Series(
            dict(nx.eigenvector_centrality(graph, max_iter=1000)),
            name='eigenvector_centrality'
        )
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality did not converge")

    logger.info(f"Centrality measures computed for {graph.number_of_nodes()} nodes")
    return results


def community_detection(graph, method: str = 'louvain') -> Dict[str, Any]:
    """
    Detect communities in a NetworkX graph.

    Args:
        graph: NetworkX graph object.
        method: Detection algorithm ('louvain' or 'greedy').

    Returns:
        Dict with 'communities' mapping, 'n_communities', 'modularity'.
    """
    import networkx as nx

    if method == 'louvain':
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(graph)
            modularity = community_louvain.modularity(partition, graph)
        except ImportError:
            # Fallback to networkx greedy modularity
            method = 'greedy'

    if method == 'greedy':
        communities_gen = nx.community.greedy_modularity_communities(graph)
        communities_list = list(communities_gen)
        partition = {}
        for i, comm in enumerate(communities_list):
            for node in comm:
                partition[node] = i
        modularity = nx.community.modularity(graph, communities_list)

    n_communities = len(set(partition.values()))

    logger.info(f"Community detection ({method}): {n_communities} communities, modularity={modularity:.3f}")
    return {
        'communities': partition,
        'n_communities': n_communities,
        'modularity': float(modularity),
        'method': method
    }


def network_clustering_coefficient(graph) -> Dict[str, Any]:
    """
    Compute clustering coefficient and small-world metrics.

    Args:
        graph: NetworkX graph object.

    Returns:
        Dict with 'average_clustering', 'transitivity', per-node 'clustering'.
    """
    import networkx as nx

    clustering = nx.clustering(graph)
    avg_clustering = nx.average_clustering(graph)
    transitivity = nx.transitivity(graph)

    result = {
        'clustering_per_node': pd.Series(clustering, name='clustering_coefficient'),
        'average_clustering': float(avg_clustering),
        'transitivity': float(transitivity),
    }

    # Small-world check: compare to random graph
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n > 1 and m > 0:
        k_mean = 2 * m / n
        if k_mean > 0:
            c_random = k_mean / n
            result['small_world_ratio'] = float(avg_clustering / c_random) if c_random > 0 else None
        else:
            result['small_world_ratio'] = None
    else:
        result['small_world_ratio'] = None

    logger.info(f"Network clustering: avg={avg_clustering:.3f}, transitivity={transitivity:.3f}")
    return result
