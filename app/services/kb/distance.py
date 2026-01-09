"""
Regime distance z-score computation for the Trading Knowledge Base.

Implements standardized Euclidean distance (diagonal Mahalanobis) for
comparing current market regime features against historical neighbor
distributions. Uses robust statistics (median/MAD) and shrinkage blending.

Key formulas:
    d(x, y) = sqrt(sum_i ((x_i - y_i)^2 / (sigma_i^2 + epsilon)))
    z = (d_now - mu) / (sigma + epsilon)
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# Epsilon to prevent division by zero
EPSILON: float = 1e-9

# MAD to standard deviation conversion factor (for Gaussian)
MAD_SCALE: float = 1.4826


@dataclass
class DistributionStats:
    """Statistics describing a distance distribution."""

    mu: float = 0.0  # Central tendency (median)
    sigma: float = 0.0  # Dispersion (scaled MAD)


@dataclass
class DistanceResult:
    """
    Result of regime distance z-score computation.

    Attributes:
        z_score: Distance z-score (None if computation not possible)
        distance: Raw standardized distance to neighborhood centroid
        mu: Baseline distance (median of neighbor distances)
        sigma: Distance dispersion (scaled MAD)
        n_neighbors: Number of neighbors used
        baseline: Type of baseline used ("composite", "neighbors_only", "none")
        missing: List of reasons if z_score is None
    """

    z_score: Optional[float] = None
    distance: Optional[float] = None
    mu: float = 0.0
    sigma: float = 0.0
    n_neighbors: int = 0
    baseline: str = "none"
    missing: list[str] = field(default_factory=list)


def compute_standardized_distance(
    x: dict[str, float],
    y: dict[str, float],
    var: dict[str, float],
) -> float:
    """
    Compute standardized Euclidean distance between two feature vectors.

    This is a diagonal Mahalanobis distance where each feature dimension
    is scaled by its variance.

    d(x, y) = sqrt(sum_i ((x_i - y_i)^2 / (var_i + epsilon)))

    Args:
        x: First feature vector as dict (feature_name -> value)
        y: Second feature vector as dict (feature_name -> value)
        var: Variance per feature (feature_name -> variance)

    Returns:
        Standardized Euclidean distance (non-negative)
    """
    # Get common keys
    common_keys = set(x.keys()) & set(y.keys())
    if not common_keys:
        return 0.0

    total = 0.0
    for k in common_keys:
        diff = x[k] - y[k]
        v = var.get(k, EPSILON)  # Use epsilon if missing
        # Ensure variance is positive
        v = max(v, EPSILON)
        total += (diff * diff) / v

    return math.sqrt(total)


def compute_distance_distribution(
    distances: list[float],
) -> DistributionStats:
    """
    Compute robust distribution statistics for a list of distances.

    Uses median and MAD (Median Absolute Deviation) which are robust
    to outliers, unlike mean and standard deviation.

    Args:
        distances: List of distance values

    Returns:
        DistributionStats with mu (median) and sigma (scaled MAD)
    """
    if not distances:
        return DistributionStats(mu=0.0, sigma=0.0)

    n = len(distances)
    if n == 1:
        return DistributionStats(mu=distances[0], sigma=0.0)

    # Sort for median calculation
    sorted_d = sorted(distances)

    # Compute median
    if n % 2 == 0:
        median = (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2
    else:
        median = sorted_d[n // 2]

    # Compute MAD (Median Absolute Deviation)
    abs_devs = [abs(d - median) for d in distances]
    sorted_devs = sorted(abs_devs)

    if n % 2 == 0:
        mad = (sorted_devs[n // 2 - 1] + sorted_devs[n // 2]) / 2
    else:
        mad = sorted_devs[n // 2]

    # Scale MAD to estimate standard deviation (for Gaussian distribution)
    sigma = mad * MAD_SCALE

    return DistributionStats(mu=median, sigma=sigma)


def _compute_centroid(
    neighbors: list[dict[str, float]],
) -> dict[str, float]:
    """
    Compute the centroid (mean) of neighbor feature vectors.

    Args:
        neighbors: List of feature dicts

    Returns:
        Centroid feature dict
    """
    if not neighbors:
        return {}

    # Collect all keys
    all_keys = set()
    for n in neighbors:
        all_keys.update(n.keys())

    centroid = {}
    for k in all_keys:
        values = [n.get(k, 0.0) for n in neighbors if k in n]
        if values:
            centroid[k] = sum(values) / len(values)

    return centroid


def _compute_observed_variance(
    neighbors: list[dict[str, float]],
    centroid: dict[str, float],
) -> dict[str, float]:
    """
    Compute observed variance from neighbors.

    Args:
        neighbors: List of feature dicts
        centroid: Centroid feature dict

    Returns:
        Variance per feature
    """
    if len(neighbors) < 2:
        return {}

    observed_var = {}
    for k in centroid.keys():
        values = [n.get(k, centroid[k]) for n in neighbors if k in n]
        if len(values) >= 2:
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            observed_var[k] = var

    return observed_var


def _shrink_variance(
    cluster_var: dict[str, float],
    observed_var: dict[str, float],
    n_neighbors: int,
    shrinkage_c: float,
) -> dict[str, float]:
    """
    Shrink observed variance toward cluster prior.

    shrunk = (k / (k + c)) * observed + (c / (k + c)) * cluster

    Args:
        cluster_var: Cluster-level variance (prior)
        observed_var: Variance from neighbors
        n_neighbors: Number of neighbors (k)
        shrinkage_c: Shrinkage constant

    Returns:
        Blended variance dict
    """
    if not cluster_var:
        return observed_var

    all_keys = set(cluster_var.keys()) | set(observed_var.keys())
    shrunk = {}

    k = n_neighbors
    c = shrinkage_c
    weight_obs = k / (k + c)
    weight_prior = c / (k + c)

    for key in all_keys:
        obs = observed_var.get(key, 0.0)
        prior = cluster_var.get(key, 0.0)
        # If only one source available, use it
        if key not in observed_var:
            shrunk[key] = prior
        elif key not in cluster_var:
            shrunk[key] = obs
        else:
            shrunk[key] = weight_obs * obs + weight_prior * prior

    return shrunk


def compute_regime_distance_z(
    current: dict[str, float],
    neighbors: list[dict[str, float]],
    cluster_var: Optional[dict[str, float]] = None,
    shrinkage_c: float = 10.0,
) -> DistanceResult:
    """
    Compute distance z-score for current regime vs neighbors.

    The z-score measures how unusual the current regime is relative
    to its neighborhood. High z-scores indicate outlier regimes that
    may warrant lower confidence in recommendations.

    Algorithm:
    1. Compute centroid of neighbors
    2. Compute distance from current to centroid
    3. Compute distances from each neighbor to centroid
    4. Build robust distribution (median/MAD) of neighbor distances
    5. Compute z = (d_current - mu) / (sigma + epsilon)

    Variance handling:
    - If cluster_var provided: blend with observed variance via shrinkage
    - If no cluster_var: use observed variance from neighbors only

    Args:
        current: Current regime features as dict
        neighbors: List of neighbor regime feature dicts
        cluster_var: Optional cluster-level variance per feature
        shrinkage_c: Shrinkage constant (higher = more prior weight)

    Returns:
        DistanceResult with z-score and diagnostic info
    """
    # Handle empty neighbors
    if not neighbors:
        return DistanceResult(
            z_score=None,
            distance=None,
            n_neighbors=0,
            baseline="none",
            missing=["no_neighbors"],
        )

    n_neighbors = len(neighbors)

    # Compute neighborhood centroid
    centroid = _compute_centroid(neighbors)

    # Compute observed variance from neighbors
    observed_var = _compute_observed_variance(neighbors, centroid)

    # Determine variance to use
    if cluster_var is not None:
        # Blend cluster prior with observed via shrinkage
        variance = _shrink_variance(
            cluster_var, observed_var, n_neighbors, shrinkage_c
        )
        baseline = "composite"
    else:
        # Use only observed variance
        variance = observed_var if observed_var else {k: EPSILON for k in current}
        baseline = "neighbors_only"

    # Ensure we have some variance to work with
    if not variance:
        variance = {k: EPSILON for k in current}

    # Distance from current to centroid
    d_current = compute_standardized_distance(current, centroid, variance)

    # Distances from each neighbor to centroid
    neighbor_distances = [
        compute_standardized_distance(n, centroid, variance) for n in neighbors
    ]

    # Compute robust distribution of neighbor distances
    dist_stats = compute_distance_distribution(neighbor_distances)

    # Compute z-score
    # When sigma is very small (low variance in neighbor distances), we need
    # a sensible floor to avoid z-score explosion. In this case, we consider
    # any distance within [0, 2*mu] as "normal" (z within -1 to +1).
    if dist_stats.sigma > EPSILON:
        sigma = dist_stats.sigma
    elif dist_stats.mu > EPSILON:
        # Use mu as sigma floor - this means distances from 0 to 2*mu
        # will have z-scores in [-1, +1] range
        sigma = dist_stats.mu
    else:
        # Both mu and sigma are ~0 means all distances are ~0
        # In this case, d_current should also be ~0, giving z ~0
        sigma = EPSILON

    z_score = (d_current - dist_stats.mu) / sigma

    return DistanceResult(
        z_score=z_score,
        distance=d_current,
        mu=dist_stats.mu,
        sigma=dist_stats.sigma,
        n_neighbors=n_neighbors,
        baseline=baseline,
        missing=[],
    )
