import logging
from inspect import Parameter, isclass, signature

import numpy as np
import torch
from torch import Tensor, linalg
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to specify if the
    estimator is fitted or not.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Raises
    ------
    TypeError
        If the estimator is a class or not an estimator instance

    NotFittedError
        If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        fitted = all_or_any([hasattr(estimator, attr) for attr in attributes])
    elif hasattr(estimator, "__sklearn_is_fitted__"):
        fitted = estimator.__sklearn_is_fitted__()
    else:
        fitted = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]

    if not fitted:
        raise


def _check_shape(param, param_shape, name):
    """Validate the shape of the input parameter 'param'.

    Parameters
    ----------
    param : array

    param_shape : tuple

    name : str
    """
    param = np.array(param)
    if param.shape != param_shape:
        raise ValueError(
            "The parameter '%s' should have the shape of %s, but got %s" % (name, param_shape, param.shape)
        )


def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like of shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(weights, (n_components,), "weights")

    # check range
    if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
        raise ValueError(
            "The parameter 'weights' should be in the range "
            "[0, 1], but got max value %.5f, min value %.5f" % (np.min(weights), np.max(weights))
        )

    # check normalization
    if not np.allclose(np.abs(1.0 - np.sum(weights)), 0.0):
        raise ValueError("The parameter 'weights' should be normalized, but got sum(weights) = %.5f" % np.sum(weights))
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), "means")
    return means


def _check_precision_positivity(precision, covariance_type):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError("'%s precision' should be positive" % covariance_type)


def _check_precision_matrix(precision, covariance_type):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (np.allclose(precision, precision.T) and np.all(linalg.eigvalsh(precision) > 0.0)):
        raise ValueError("'%s precision' should be symmetric, positive-definite" % covariance_type)


def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)


def _check_precisions(precisions, covariance_type, n_components, n_features):
    """Validate user provided precisions.

    Parameters
    ----------
    precisions : array-like
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : str

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(
        precisions,
        dtype=[np.float64, np.float32],
        ensure_2d=False,
        allow_nd=covariance_type == "full",
    )

    precisions_shape = {
        "full": (n_components, n_features, n_features),
        "tied": (n_features, n_features),
        "diag": (n_components, n_features),
        "spherical": (n_components,),
    }
    _check_shape(precisions, precisions_shape[covariance_type], "%s precision" % covariance_type)

    _check_precisions = {
        "full": _check_precisions_full,
        "tied": _check_precision_matrix,
        "diag": _check_precision_positivity,
        "spherical": _check_precision_positivity,
    }
    _check_precisions[covariance_type](precisions, covariance_type)
    return precisions


def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = torch.empty((n_components, n_features, n_features), dtype=X.dtype)
    for k in range(n_components):
        diff = X - means[k]
        cov = torch.mm(resp[:, k] * diff.T, diff) / nk[k]
        trace = cov.diag().sum()
        covariances[k] = cov.cpu()
        covariances[k].flatten()[:: n_features + 1] += reg_covar * trace.cpu()  # add reg_covar to the diagonal
    return covariances


def _estimate_gaussian_covariances_tied(resp, X, nk, means, reg_covar):
    """Estimate the tied covariance matrix.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariance : array, shape (n_features, n_features)
        The tied covariance matrix of the components.
    """
    avg_X2 = torch.mm(X.T, X)
    avg_means2 = torch.mm(nk * means.T, means)
    covariance = avg_X2 - avg_means2
    covariance /= nk.sum()
    trace = covariance.diag().sum()
    covariance.flatten()[:: len(covariance) + 1] += reg_covar * trace
    return covariance


def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    avg_X2 = torch.mm(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means**2
    avg_X_means = means * torch.mm(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar


def _estimate_gaussian_covariances_spherical(resp, X, nk, means, reg_covar):
    """Estimate the spherical variance values.

    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    variances : array, shape (n_components,)
        The variance values of each components.
    """
    return _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)


def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(dim=0) + 10 * torch.finfo(resp.dtype).eps
    means = resp.T @ X / nk[:, np.newaxis]
    covariances = {
        "full": _estimate_gaussian_covariances_full,
        "tied": _estimate_gaussian_covariances_tied,
        "diag": _estimate_gaussian_covariances_diag,
        "spherical": _estimate_gaussian_covariances_spherical,
    }[covariance_type](resp, X, nk, means, reg_covar)
    return nk, means, covariances


def _compute_precision_cholesky(covariances, covariance_type, device="cpu"):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """

    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        precisions_chol = torch.empty((n_components, n_features, n_features), dtype=covariances.dtype)
        for k, covariance in enumerate(covariances):
            cov_chol = linalg.cholesky(covariance.to(device), upper=False)
            precisions_chol[k] = linalg.solve_triangular(
                cov_chol, torch.eye(n_features, device=cov_chol.device), upper=False
            ).T.cpu()
    elif covariance_type == "tied":
        _, n_features = covariances.shape
        cov_chol = linalg.cholesky(covariances, upper=False)
        precisions_chol = linalg.solve_triangular(
            cov_chol, torch.eye(n_features, device=cov_chol.device), upper=False
        ).T
    else:
        if torch.any(torch.less_equal(covariances, 0.0)):
            raise ValueError
        precisions_chol = 1.0 / torch.sqrt(covariances)
    return precisions_chol


###############################################################################
# Gaussian mixture probability estimators
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.

    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    n_features : int
        Number of features.

    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = torch.sum(torch.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1)

    elif covariance_type == "tied":
        log_det_chol = torch.sum(torch.log(torch.diag(matrix_chol)))

    elif covariance_type == "diag":
        log_det_chol = torch.sum(torch.log(matrix_chol), dim=1)

    else:
        log_det_chol = n_features * (torch.log(matrix_chol))

    return log_det_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    covariance_type : {'full', 'tied', 'diag', 'spherical'}

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # The determinant of the precision matrix from the Cholesky decomposition
    # corresponds to the negative half of the determinant of the full precision
    # matrix.
    # In short: det(precision_chol) = - det(precision) / 2
    log_det = _compute_log_det_cholesky(precisions_chol.to(X.device), covariance_type, n_features)

    if covariance_type == "full":
        log_prob = torch.empty((n_samples, n_components), dtype=X.dtype)
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = torch.mm(X, prec_chol.to(X.device)) - torch.mm(mu.unsqueeze(0).to(X.device), prec_chol.to(X.device))
            log_prob[:, k] = torch.sum(torch.square(y), dim=1).cpu()

    elif covariance_type == "tied":
        log_prob = torch.empty((n_samples, n_components), dtype=X.dtype)
        for k, mu in enumerate(means):
            y = torch.mm(X, precisions_chol.to(X.device)) - torch.mm(
                mu.unsqueeze(0).to(X.device), precisions_chol.to(X.device)
            )
            log_prob[:, k] = torch.sum(torch.square(y), dim=1).cpu()

    elif covariance_type == "diag":
        precisions = precisions_chol**2
        log_prob = (
            torch.sum((means.to(X.device) ** 2 * precisions.to(X.device)), 1)
            - 2.0 * torch.mm(X, (means.to(X.device) * precisions.to(X.device)).T)
            + torch.mm(X**2, precisions.to(X.device).T)
        )

    elif covariance_type == "spherical":
        raise NotImplementedError("spherical covariance is not implemented yet")
        precisions = precisions_chol**2
        log_prob = (
            torch.sum(means**2, 1) * precisions
            - 2 * torch.mm(X, means.T * precisions)
            + torch.outer(row_norms(X, squared=True), precisions)
        )
    # Since we are using the precision of the Cholesky decomposition,
    # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
    return -0.5 * (n_features * torch.log(2 * torch.tensor(torch.pi)) + log_prob) + log_det.to(log_prob.device)


class GaussianMixture:
    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-4,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = X.shape

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init, self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init, self.n_components, n_features)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(
                self.precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = _estimate_gaussian_parameters(X, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(covariances, self.covariance_type, device=X.device)
        elif self.covariance_type == "full":
            self.precisions_cholesky_ = torch.tensor(
                [linalg.cholesky(prec_init, upper=False) for prec_init in self.precisions_init]
            )
        elif self.covariance_type == "tied":
            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init, upper=False)
        else:
            self.precisions_cholesky_ = torch.sqrt(self.precisions_init)

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, torch.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type, device=X.device
        )

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(X, self.means_, self.precisions_cholesky_, self.covariance_type).to(X.device)

    def _estimate_log_weights(self):
        return torch.log(self.weights_)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.precisions_cholesky_,
        ) = params

        # Attributes computation
        _, n_features = self.means_.shape

        if self.covariance_type == "full":
            self.precisions_ = torch.empty(self.precisions_cholesky_.shape)
            for k, prec_chol in enumerate(self.precisions_cholesky_):
                self.precisions_[k] = torch.mm(prec_chol, prec_chol.T)

        elif self.covariance_type == "tied":
            self.precisions_ = torch.mm(self.precisions_cholesky_, self.precisions_cholesky_.T)
        else:
            self.precisions_ = self.precisions_cholesky_**2

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == "full":
            cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "diag":
            cov_params = self.n_components * n_features
        elif self.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) / 2.0
        elif self.covariance_type == "spherical":
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def _initialize_parameters(self, X):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape

        if self.init_params == "kmeans":
            raise NotImplementedError("Kmeans initialization is not implemented for GaussianMixture")
            resp = torch.zeros((n_samples, self.n_components))
            label = (
                cluster.KMeans(n_clusters=self.n_components, n_init=1, random_state=self.random_state).fit(X).labels_
            )
            resp[torch.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = torch.rand(size=(n_samples, self.n_components), device=X.device, dtype=X.dtype)
            resp /= resp.sum(dim=1)[:, np.newaxis]
        elif self.init_params == "random_from_data":
            resp = torch.zeros((n_samples, self.n_components), device=X.device, dtype=X.dtype)
            indices = torch.randperm(n_samples)
            resp[indices, :] = 1
        elif self.init_params == "k-means++":
            raise NotImplementedError("Kmeans++ initialization is not implemented for GaussianMixture")
            resp = torch.zeros((n_samples, self.n_components))
            _, indices = kmeans_plusplus(
                X,
                self.n_components,
                random_state=self.random_state,
            )
            resp[indices, torch.arange(self.n_components)] = 1
        else:
            raise ValueError("Unimplemented initialization method '%s'" % self.init_params)

        self._initialize(X, resp)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        bic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * torch.log(X.shape[0])

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            The fitted mixture.
        """
        # parameters are validated in fit_predict
        self.fit_predict(X, y)
        return self

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -torch.inf
        self.converged_ = False

        n_samples, _ = X.shape
        for init in range(n_init):
            if do_init:
                self._initialize_parameters(X)

            lower_bound = -torch.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                for n_iter in tqdm(range(1, self.max_iter + 1), desc="EM iteration"):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X)
                    self._m_step(X, log_resp)
                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                    change = lower_bound - prev_lower_bound

                    if abs(change) < self.tol:
                        self.converged_ = True
                        break

                if lower_bound > max_lower_bound or max_lower_bound == -torch.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            _logger.warn(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1),
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(dim=1)

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return torch.mean(log_prob_norm), log_resp

    def score_samples(self, X):
        """Compute the log-likelihood of each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample in `X` under the current model.
        """
        check_is_fitted(self)

        return torch.logsumexp(self._estimate_weighted_log_prob(X), dim=1)

    def score(self, X, y=None):
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of `X` under the Gaussian mixture model.
        """
        return self.score_samples(X).mean()

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        check_is_fitted(self)
        return self._estimate_weighted_log_prob(X).argmax(dim=1)

    def predict_proba(self, X):
        """Evaluate the components' density for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Density of each Gaussian component for each sample in X.
        """
        check_is_fitted(self)
        _, log_resp = self._estimate_log_prob_resp(X)
        return torch.exp(log_resp)

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        y : array, shape (nsamples,)
            Component labels.
        """
        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        _, n_features = self.means_.shape
        n_samples_comp = np.random.multinomial(n_samples, self.weights_.cpu().numpy()).tolist()

        if self.covariance_type == "full":
            X = torch.vstack(
                [
                    MultivariateNormal(loc=mean, covariance_matrix=covariance.to(mean.device)).sample((sample,))
                    for (mean, covariance, sample) in zip(self.means_, self.covariances_, n_samples_comp)
                ]
            )
        elif self.covariance_type == "tied":
            X = torch.vstack(
                [
                    MultivariateNormal(loc=mean, covariance_matrix=self.covariances_).sample((sample,))
                    for (mean, sample) in zip(self.means_, n_samples_comp)
                ]
            )
        else:
            X = torch.vstack(
                [
                    mean + torch.randn(size=(sample, n_features), device=covariance.device) * torch.sqrt(covariance)
                    for (mean, covariance, sample) in zip(self.means_, self.covariances_, n_samples_comp)
                ]
            )

        y = torch.concatenate(
            [torch.ones(sample, dtype=torch.int, device=X.device) * j for j, sample in enumerate(n_samples_comp)]
        )

        return (X, y)

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights().to(X.device)

    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp


def test():
    import itertools

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = 100
    d = 2
    c = 3

    covariance_types = ["full", "tied", "diag"]
    init_params_types = ["random", "random_from_data"]
    for cov_type, init_param in itertools.product(covariance_types, init_params_types):
        print(f"Testing {cov_type} {init_param}...")
        gmm = GaussianMixture(n_components=c, covariance_type=cov_type, max_iter=100, tol=1e-3, init_params=init_param)
        X = torch.randn(n, d, dtype=torch.float32).to(device)
        gmm.fit(X)
        params = gmm._get_parameters()
        for p in params:
            print(p.device, p.shape, p.dtype)
        gmm.predict(X)
        gmm.predict_proba(X)
        gmm.score(X)
        scores = gmm.score_samples(X)
        gmm.sample(10)

        assert gmm.means_.shape == (c, d)
        assert scores.shape == (100,)


def benchmark_accelerated_gmm():
    import time

    time_cpu = []
    time_cuda = []
    configs = []
    for n in [10_000, 50_000, 200_000]:
        for d in [512, 768, 1024, 2048]:
            for c in [10, 100, 1000]:
                try:
                    configs.append((n, d, c))
                    print(f"n={n}, d={d}, c={c}", end=": ")
                    if len(time_cpu) == 0 or time_cpu[-1] < 10:
                        X = torch.randn(n, d, dtype=torch.float32)
                        gmm = GaussianMixture(
                            n_components=c,
                            covariance_type="full",
                            max_iter=100,
                            tol=1e-3,
                            init_params="random_from_data",
                        )
                        start = time.time()
                        gmm.fit(X)
                        end = time.time()
                        time_cpu.append(end - start)
                        print(f"CPU time: {end - start:.4f}s", end=", ")

                    X = torch.randn(n, d, dtype=torch.float32, device="cuda")
                    gmm = GaussianMixture(
                        n_components=c, covariance_type="full", max_iter=100, tol=1e-1, init_params="random_from_data"
                    )
                    start_cuda = time.time()
                    gmm.fit(X)
                    end_cuda = time.time()
                    time_cuda.append(end_cuda - start_cuda)
                    print(f"CUDA time: {end_cuda - start_cuda:.4f}s")
                except RuntimeError as e:
                    print(e)
                    continue

    print("CPU times:")
    for (n, d, c), t in zip(configs, time_cpu):
        print(f"n={n}, d={d}, c={c}: {t:.4f}s")

    # sort configs and time_gpu
    configs, time_cuda = zip(*sorted(zip(configs, time_cuda)))
    print("CUDA times:")
    for (n, d, c), t in zip(configs, time_cuda):
        print(f"n={n}, d={d}, c={c}: {t:.4f}s")


if __name__ == "__main__":
    test()
    benchmark_accelerated_gmm()
