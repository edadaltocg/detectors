import numpy as np
from scipy import stats


def p_value_fn(test_statistic: np.ndarray, X: np.ndarray, w=None):
    """Compute the p-value of a test statistic given a sample X.

    Args:
        test_statistic (np.ndarray): test statistic (n,m)
        X (np.ndarray): sample (N,m)

    Returns:
        np.ndarray: p-values (n,m)
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(test_statistic.shape) == 1:
        test_statistic = test_statistic.reshape(-1, 1)
    mult_factor_min = np.where(X.min(0) > 0, np.array(1 / len(X)), np.array(len(X)))
    mult_factor_max = np.where(X.max(0) > 0, np.array(len(X)), np.array(1 / len(X)))
    lower_bound = X.min(0) * mult_factor_min
    upper_bound = X.max(0) * mult_factor_max
    X = np.concatenate((lower_bound.reshape(1, -1), X, upper_bound.reshape(1, -1)), axis=0)
    X = np.sort(X, axis=0)
    y_ecdf = np.concatenate([np.arange(1, X.shape[0] + 1).reshape(-1, 1) / X.shape[0]] * X.shape[1], axis=1)
    if w is not None:
        y_ecdf = y_ecdf * w.reshape(1, -1)
    return np.concatenate(list(map(lambda xx: np.interp(*xx).reshape(-1, 1), zip(test_statistic.T, X.T, y_ecdf.T))), 1)


def fisher_method(p_values: np.ndarray):
    """Combine p-values using Fisher's method

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    tau = -2 * np.sum(np.log(p_values), axis=1).reshape(-1, 1)
    group_p_value = p_value_fn(tau, np.random.chisquare(2 * p_values.shape[1], (1000, 1)))
    # or
    # group_p_value = stats.chi2.cdf(tau, 2 * p_values.shape[1])
    return group_p_value


def fisher_tau_method(p_values: np.ndarray):
    """Combine p-values using Fisher's method

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    tau = -2 * np.sum(np.log(p_values), axis=1)
    return tau


def stouffer_method(p_values: np.ndarray):
    """Combine p-values using Stouffer's method

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    z = np.sum(stats.norm.ppf(p_values), axis=1).reshape(-1, 1) / np.sqrt(p_values.shape[1])
    group_p_value = p_value_fn(z, np.random.normal(size=(1000, 1)))
    # or
    # group_p_value = stats.norm.cdf(z)
    return group_p_value


def stouffer_tau_method(p_values: np.ndarray):
    """Combine p-values using Stouffer's method

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    z = np.sum(stats.norm.ppf(p_values), axis=1) / np.sqrt(p_values.shape[1])
    # max that is not inf
    max_not_inf = np.max(z[np.isfinite(z)])
    min_not_inf = np.min(z[np.isfinite(z)])
    # replace inf with max or min
    z = np.where(np.isposinf(z), max_not_inf, z)
    z = np.where(np.isneginf(z), min_not_inf, z)
    return z


def tippet_tau_method(p_values: np.ndarray):
    """Combine p-values using Tippet's method

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    tau = np.min(p_values, axis=1)
    return tau


def wilkinson_tau_method(p_values: np.ndarray):
    """Combine p-values using Wilkinson's method

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    tau = np.max(p_values, axis=1)
    return tau


def edgington_tau_method(p_values: np.ndarray):
    """Combine p-values using Edington's method

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    tau = np.sum(p_values, axis=1)
    return tau


def pearson_tau_method(p_values: np.ndarray):
    """Combine p-values using Pearson's method

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    tau = 2 * np.sum(np.log(1 - p_values + 1e-6), axis=1)
    return tau


def simes_tau_method(p_values: np.ndarray):
    """Combine p-values using Simes' method

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    tau = np.min(np.sort(p_values, axis=1) / np.arange(1, p_values.shape[1] + 1) * p_values.shape[1], 1)
    return tau


def geometric_mean_tau_method(p_values: np.ndarray):
    """Combine p-values using geometric mean

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    tau = np.prod(p_values, axis=1) ** (1 / p_values.shape[1])
    return tau

def rho(p_values):
    k = p_values.shape[1]
    phi = stats.norm.ppf(p_values)
    return 1 - (1 / (k - 1)) * np.sum((phi - np.mean(phi, axis=1, keepdims=True)) ** 2, axis=1)


def hartung(p_values, r):
    k = p_values.shape[1]
    t = stats.norm.ppf(p_values)
    return np.sum(t, axis=1) / np.sqrt((1 - r) * k + r * k**2

def get_combine_p_values_fn(method_name: str):
    method_name = method_name.lower()
    if method_name == "fisher":
        return fisher_tau_method
    elif method_name == "stouffer":
        return stouffer_tau_method
    elif method_name == "tippet":
        return tippet_tau_method
    elif method_name == "wilkinson":
        return wilkinson_tau_method
    elif method_name == "edgington":
        return edgington_tau_method
    elif method_name == "pearson":
        return pearson_tau_method
    elif method_name == "simes":
        return simes_tau_method
    elif method_name == "geometric_mean":
        return geometric_mean_tau_method
    else:
        raise NotImplementedError(f"method {method_name} not implemented")


ensemble_names = ["fisher", "stouffer", "tippet", "wilkinson", "edgington", "pearson", "simes", "geometric_mean"]