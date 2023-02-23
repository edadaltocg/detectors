r"""Pelt.

from `https://github.com/deepcharles/ruptures/blob/master/src/ruptures/detection/pelt.py`"""
from math import floor
from .base import BaseCost, BaseEstimator


class CostL2(BaseCost):
    r"""Least squared deviation."""

    model = "l2"

    def __init__(self):
        """Initialize the object."""
        self.signal = None
        self.min_size = 1

    def fit(self, signal) -> "CostL2":
        """Set parameters of the instance.
        Args:
            signal (array): array of shape (n_samples,) or (n_samples, n_features)
        Returns:
            self
        """
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal

        return self

    def error(self, start, end) -> float:
        """Return the approximation cost on the segment [start:end].
        Args:
            start (int): start of the segment
            end (int): end of the segment
        Returns:
            segment cost
        Raises:
            NotEnoughPoints: when the segment is too short (less than `min_size` samples).
        """

        return self.signal[start:end].var(axis=0).sum() * (end - start)


class PeltL2(BaseEstimator):
    """Penalized change point detection.

    For a given model and penalty level, computes the segmentation which
    minimizes the constrained sum of approximation errors.
    """

    def __init__(self, min_size=2, jump=5):
        """Initialize a Pelt instance.

        Args:
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
        """
        self.cost = CostL2()
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None

    def _seg(self, pen):
        """Computes the segmentation for a given penalty using PELT (or a list
        of penalties).

        Args:
            penalty (float): penalty value

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """

        # initialization
        # partitions[t] contains the optimal partition of signal[0:t]
        partitions = dict()  # this dict will be recursively filled
        partitions[0] = {(0, 0): 0}
        admissible = []

        # Recursion
        ind = [k for k in range(0, self.n_samples, self.jump) if k >= self.min_size]
        ind += [self.n_samples]
        for bkp in ind:
            # adding a point to the admissible set from the previous loop.
            new_adm_pt = floor((bkp - self.min_size) / self.jump)
            new_adm_pt *= self.jump
            admissible.append(new_adm_pt)

            subproblems = list()
            for t in admissible:
                # left partition
                try:
                    tmp_partition = partitions[t].copy()
                except KeyError:  # no partition of 0:t exists
                    continue
                # we update with the right partition
                tmp_partition.update({(t, bkp): self.cost.error(t, bkp) + pen})
                subproblems.append(tmp_partition)

            # finding the optimal partition
            partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
            # trimming the admissible set
            admissible = [
                t
                for t, partition in zip(admissible, subproblems)
                if sum(partition.values()) <= sum(partitions[bkp].values()) + pen
            ]

        best_partition = partitions[self.n_samples]
        del best_partition[(0, 0)]
        return best_partition

    def fit(self, signal):
        """Set params.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update params
        self.cost.fit(signal)
        if signal.ndim == 1:
            (n_samples,) = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        return self

    def predict(self, pen=1):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.pelt.Pelt.fit].

        Args:
            pen (float): penalty value (>0)

        Raises:
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list: sorted list of breakpoints
        """
        partition = self._seg(pen)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, pen=1):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(pen)
