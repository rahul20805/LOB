"""
Statistical Significance Testing Engine for Quantitative Strategy Evaluation.
Implements:
1. Diebold-Mariano test for forecast comparison
2. McNemar's test for paired classification accuracy comparison
3. Wilcoxon signed-rank test for paired trading returns
4. Non-parametric bootstrap confidence intervals (95% CI)
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats


def compute_bootstrap_ci(
    values: np.ndarray,
    stat_func: Any = np.mean,
    n_bootstraps: int = 1000,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Computes empirical bootstrap confidence intervals (mean, lower_ci, upper_ci).
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0
        
    boot_stats = []
    n = len(values)
    for _ in range(n_bootstraps):
        sample = np.random.choice(values, size=n, replace=True)
        boot_stats.append(stat_func(sample))
        
    boot_stats = np.sort(boot_stats)
    lower = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    point_est = float(stat_func(values))
    return point_est, lower, upper


def diebold_mariano_test(
    loss_model1: np.ndarray,
    loss_model2: np.ndarray,
    h: int = 1
) -> Dict[str, float]:
    """
    Diebold-Mariano test comparing forecast accuracy loss series.
    """
    d = loss_model1 - loss_model2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    
    # Auto-covariance adjustment for horizon h
    gamma_0 = var_d
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[:-k], d[k:])[0, 1] if len(d) > k else 0.0
        gamma_sum += (1 - k / h) * gamma_k
        
    lr_var = (gamma_0 + 2 * gamma_sum) / len(d)
    if lr_var <= 0:
        lr_var = 1e-8
        
    dm_stat = mean_d / np.sqrt(lr_var)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    
    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "is_significant_5pct": bool(p_value < 0.05)
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> Dict[str, float]:
    """
    McNemar's test for comparing paired classification outputs.
    """
    correct1 = y_pred1 == y_true
    correct2 = y_pred2 == y_true
    
    # b: Model 1 correct, Model 2 incorrect
    b = np.sum(correct1 & ~correct2)
    # c: Model 1 incorrect, Model 2 correct
    c = np.sum(~correct1 & correct2)
    
    if (b + c) == 0:
        return {"chi2_stat": 0.0, "p_value": 1.0, "is_significant_5pct": False}
        
    chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_value = 1.0 - stats.chi2.cdf(chi2, df=1)
    
    return {
        "chi2_stat": float(chi2),
        "p_value": float(p_value),
        "is_significant_5pct": bool(p_value < 0.05)
    }


def wilcoxon_signed_rank_test(
    returns_model1: np.ndarray,
    returns_model2: np.ndarray
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test comparing paired net return distributions.
    """
    diff = returns_model1 - returns_model2
    diff = diff[diff != 0]
    if len(diff) < 10:
        return {"statistic": 0.0, "p_value": 1.0, "is_significant_5pct": False}
        
    res = stats.wilcoxon(diff)
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "is_significant_5pct": bool(res.pvalue < 0.05)
    }
