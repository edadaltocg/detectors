import numpy as np
import optuna
import scipy
import scipy.optimize
import torch

optuna.logging.set_verbosity(optuna.logging.WARNING)


def invert(x):
    sign = np.sign(x[0])
    return sign * 1 / (x + 1e-7)


def c_in_fn(lbd, pi):
    return lbd * pi


def c_out_fn(lbd, pi, c_fn):
    return c_fn - lbd * (1 - pi)


def thr_fn(lbd, pi, c_fn):
    return 1 - 2 * c_in_fn(lbd, pi) - c_out_fn(lbd, pi, c_fn)


def plugin_bb(
    scores_sc: np.ndarray,
    scores_ood: np.ndarray,
    labels: np.ndarray,
    c_fn=0.75,
    sort=True,
    n=101,
    backend: str = "brute",
    pi=None,
):
    """
    labels take values in {0, 1}
        - 0: in_distribution, correctly classified
        - 1: misclassified, or out-of-distribution

    r_bb == 1 => reject (misclassified or OOD)
    """
    # build rejector: need to compute c_in and c_out
    # c_in = lbd * pi
    # c_out = c_fn - lbd * (1 - pi)
    # thr = 1 - 2 * c_in - c_out
    # pi is asy to estimate
    # lbd need to solve the lagragnian or solve Pr(r(x)=1) <= b_rej

    def r_bb_fn(c_in, c_out, thr):
        return (1 - c_in - c_out) * scores_sc + c_out * invert(scores_ood) < thr

    @torch.no_grad()
    def r_bb_fn_parallel(c_in, c_out, thr):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        c_in = torch.from_numpy(c_in).float().to(device)
        c_out = torch.from_numpy(c_out).float().to(device)
        thr = torch.from_numpy(thr).float().to(device)
        scores_sc_ = torch.from_numpy(scores_sc).float().to(device)
        scores_ood_ = torch.from_numpy(scores_ood).float().to(device)
        calc = (
            (1 - c_in - c_out).T @ scores_sc_.reshape(1, -1) + c_out.T @ (scores_ood_.reshape(1, -1) ** -1) < thr.T
        ).cpu()
        return calc.numpy()

    if pi is None:
        pi = sum(labels == 0) / (sum(labels == 0) + sum(labels == 1))

    # find lambda such that torch.mean(r_bb_fn(c_in, c_out, thr)) <= b_rej
    low = min(0, c_fn / (1 - pi), (c_fn - 1) / (1 - pi), 1 / pi) - 1
    high = max((c_fn - 1) / (1 - pi), 1 / pi, low, c_fn / (1 - pi)) + 1

    risks = []
    coverages = []
    thrs = []
    for brej in np.linspace(0, 1, n):

        def obj_optuna(trial: optuna.Trial):
            lbd = trial.suggest_float("lbd", low, high)
            cin = c_in_fn(lbd, pi)
            cout = c_out_fn(lbd, pi, c_fn)
            thr = thr_fn(lbd, pi, c_fn)
            return (np.mean(r_bb_fn(cin, cout, thr)) - brej) ** 2

        def obj_scipy(lbd):
            cin = c_in_fn(lbd, pi)
            cout = c_out_fn(lbd, pi, c_fn)
            thr = thr_fn(lbd, pi, c_fn)
            return (np.mean(r_bb_fn(cin, cout, thr)) - brej) ** 2

        def obj_scipy_parallel(lbds):
            lbds = lbds.reshape(1, -1)
            cins = c_in_fn(lbds, pi)
            couts = c_out_fn(lbds, pi, c_fn)
            thrs = thr_fn(lbds, pi, c_fn)
            rbbs = r_bb_fn_parallel(cins, couts, thrs)  # N_lbd x N_scores
            return (np.mean(rbbs, 1) - brej) ** 2

        if backend == "optuna":
            study = optuna.create_study()
            study.optimize(obj_optuna, n_trials=100)
            lbd_star = study.best_params["lbd"]
        elif backend == "scipy":
            res = scipy.optimize.minimize_scalar(obj_scipy, method="brent", options={"xtol": 1e-5})
            lbd_star = res.x
        elif backend == "parallel":
            x = np.linspace(low, high, 1000)
            y = obj_scipy_parallel(x)
            lbd_star = x[np.argmin(y)]
        elif backend == "brute":
            x = np.linspace(low, high, 1000)
            y = np.array([obj_scipy(xx) for xx in x])
            lbd_star = x[np.argmin(y)]
        else:
            raise ValueError("backend must be one of ['optuna', 'scipy']")

        reject_idx = r_bb_fn(c_in_fn(lbd_star, pi), c_out_fn(lbd_star, pi, c_fn), thr_fn(lbd_star, pi, c_fn))
        covered_idx = reject_idx == 0
        coverage = covered_idx.mean()
        risk = labels[covered_idx].sum() / (covered_idx.sum() + 1e-7)
        thr = thr_fn(lbd_star, pi, c_fn)

        thrs.append(thr)
        coverages.append(coverage)
        risks.append(risk)

    thrs, coverages, risks = np.array(thrs), np.array(coverages), np.array(risks)

    # sort by coverages
    if sort:
        sorted_idx = np.argsort(coverages).astype(int)
        risks = risks[sorted_idx]
        coverages = coverages[sorted_idx]
        thrs = thrs[sorted_idx]

    return risks, coverages, thrs


def plugin_bb_wrapper(
    scores_sc_in,
    scores_sc_out,
    scores_ood_in,
    scores_ood_out,
    c_fn=0.75,
    sort=True,
    n=101,
    backend: str = "brute",
    pi=None,
):
    scores_sc = np.concatenate([scores_sc_in, scores_sc_out])
    scores_ood = np.concatenate([scores_ood_in, scores_ood_out])
    labels = np.concatenate([np.zeros(len(scores_sc_in)), np.ones(len(scores_sc_out))])
    risks, coverages, thrs = plugin_bb(scores_sc, scores_ood, labels, c_fn=c_fn, sort=sort, n=n, backend=backend, pi=pi)
    # compute AURC
    aurc = np.trapz(risks, coverages)

    return {
        "aurc": float(aurc),
        "risks": risks.tolist(),
        "coverages": coverages.tolist(),
        "thrs": thrs.tolist(),
    }


def benchmark():
    import time

    def timefn(fn, *args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        end = time.time()
        print(f"Time: {end - start}")
        return res

    N = 20000
    scores_sc = np.random.rand(N)
    scores_ood = np.random.rand(N)
    labels = np.random.randint(0, 2, N)
    res1 = timefn(plugin_bb, scores_sc, scores_ood, labels, c_fn=0.75, sort=True, n=101, backend="parallel", pi=None)
    res2 = timefn(plugin_bb, scores_sc, scores_ood, labels, c_fn=0.75, sort=True, n=101, backend="brute", pi=None)
    print([a - b for a, b in zip(res1[0], res2[0])])


if __name__ == "__main__":
    # benchmark
    benchmark()
