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

# Default shrinkage constant for variance blending
DEFAULT_SHRINKAGE_C: float = 20.0


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
        distance_now: Raw standardized distance (median distance to neighbors)
        mu: Baseline distance (median of neighbor distances), None if not computed
        sigma: Distance dispersion (scaled MAD), None if not computed
        n_neighbors: Number of neighbors used
        baseline: Type of baseline used ("composite", "neighbors_only", "none")
        missing: List of reasons if z_score is None
    """

    z_score: Optional[float] = None
    distance_now: Optional[float] = None
    mu: Optional[float] = None
    sigma: Optional[float] = None
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
    current_features: dict[str, float],
    neighbor_features: list[dict[str, float]],
    cluster_var: Optional[dict[str, float]] = None,
    shrinkage_c: float = DEFAULT_SHRINKAGE_C,
    cluster_sigma_prior: Optional[float] = None,
) -> DistanceResult:
    """
    Compute distance z-score for current regime vs neighbors.

    The z-score measures how unusual the current regime is relative
    to its neighborhood. High z-scores indicate outlier regimes that
    may warrant lower confidence in recommendations.

    Algorithm:
    1. Compute distance from current to EACH neighbor (using cluster_var for scaling)
    2. Build robust distribution (median/MAD) of those distances
    3. Apply shrinkage: sigma = alpha * cluster_sigma_prior + (1-alpha) * sigma_obs
    4. Compute z = (d_now - mu) / (sigma + epsilon) where d_now is median distance

    Variance handling:
    - If cluster_var provided: use for scaling distances (blend with observed)
    - If no cluster_var: use observed variance from neighbors only

    Sigma shrinkage:
    - If cluster_sigma_prior provided: blend observed sigma toward prior
    - alpha = shrinkage_c / (n_neighbors + shrinkage_c)

    Args:
        current_features: Current regime features as dict
        neighbor_features: List of neighbor regime feature dicts
        cluster_var: Optional cluster-level variance per feature (for scaling)
        shrinkage_c: Shrinkage constant (higher = more prior weight)
        cluster_sigma_prior: Optional prior sigma for distribution shrinkage

    Returns:
        DistanceResult with z-score and diagnostic info
    """
    # Handle empty neighbors
    if not neighbor_features:
        return DistanceResult(
            z_score=None,
            distance_now=None,
            mu=None,
            sigma=None,
            n_neighbors=0,
            baseline="none",
            missing=["no_neighbors"],
        )

    n_neighbors = len(neighbor_features)

    # Compute neighborhood centroid (for variance estimation)
    centroid = _compute_centroid(neighbor_features)

    # Compute observed variance from neighbors
    observed_var = _compute_observed_variance(neighbor_features, centroid)

    # Determine variance to use for distance scaling
    if cluster_var is not None:
        # Blend cluster prior with observed via shrinkage
        variance = _shrink_variance(
            cluster_var, observed_var, n_neighbors, shrinkage_c
        )
        baseline = "composite"
    else:
        # Use only observed variance
        variance = (
            observed_var if observed_var else {k: EPSILON for k in current_features}
        )
        baseline = "neighbors_only"

    # Ensure we have some variance to work with
    if not variance:
        variance = {k: EPSILON for k in current_features}

    # Compute distance from current to EACH neighbor (spec algorithm)
    distances_to_neighbors = [
        compute_standardized_distance(current_features, n, variance)
        for n in neighbor_features
    ]

    # Compute robust distribution of distances
    dist_stats = compute_distance_distribution(distances_to_neighbors)

    # d_now is the median distance to neighbors
    d_now = dist_stats.mu

    # Observed sigma from distribution
    sigma_obs = dist_stats.sigma

    # Apply shrinkage to sigma if cluster_sigma_prior provided
    if cluster_sigma_prior is not None:
        alpha = shrinkage_c / (n_neighbors + shrinkage_c)
        sigma_final = alpha * cluster_sigma_prior + (1 - alpha) * sigma_obs
    else:
        sigma_final = sigma_obs

    # Compute z-score with floor to avoid explosion
    if sigma_final > EPSILON:
        sigma = sigma_final
    elif d_now > EPSILON:
        # Use d_now as sigma floor
        sigma = d_now
    else:
        # Both are ~0, use epsilon
        sigma = EPSILON

    z_score = (d_now - dist_stats.mu) / sigma

    return DistanceResult(
        z_score=z_score,
        distance_now=d_now,
        mu=dist_stats.mu,
        sigma=sigma_obs,
        n_neighbors=n_neighbors,
        baseline=baseline,
        missing=[],
    )
