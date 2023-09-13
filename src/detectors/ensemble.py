import numpy as np


def p_value_fn(test_statistic: np.ndarray, X: np.ndarray):
    """Compute the p-value of a test statistic given a sample X.

    Args:
        test_statistic (np.ndarray): test statistic (n,m)
        X (np.ndarray): sample (N,m)

    Returns:
        np.ndarray: p-values (n,m)
    """
    mult_factor_min = np.where(X.min(0) > 0, np.array(1 / len(X)), np.array(len(X)))
    mult_factor_max = np.where(X.max(0) > 0, np.array(len(X)), np.array(1 / len(X)))
    lower_bound = X.min(0) * mult_factor_min
    upper_bound = X.max(0) * mult_factor_max
    X = np.concatenate((lower_bound.reshape(1, -1), X, upper_bound.reshape(1, -1)), axis=0)
    X = np.sort(X, axis=0)
    y_ecdf = np.concatenate([np.arange(1, X.shape[0] + 1).reshape(-1, 1) / X.shape[0]] * X.shape[1], axis=1)
    return np.concatenate(list(map(lambda xx: np.interp(*xx).reshape(-1, 1), zip(test_statistic.T, X.T, y_ecdf.T))), 1)


# Fisher
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


# Fisher
def fisher_tau_method(p_values: np.ndarray):
    """Combine p-values using Fisher's method

    Args:
        p_values (np.ndarray): p-values (n,m)

    Returns:
        np.ndarray (n,): combined p-values
    """
    tau = -2 * np.sum(np.log(p_values), axis=1)
    return tau
