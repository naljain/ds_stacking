"""
LPV-DS in 3D Cartesian end-effector space.

    x_dot = Σ_k h_k(x) * (A_k * x + b_k)
    b_k   = -A_k * x_goal

All positions are world-frame end-effector positions.  The model is genuinely
3D, but training uses a guarded normalisation scale so the almost-constant
transport height does not let tiny z noise dominate GMM assignment.

Deploy:
    x_dot = model.predict(ee_pos)          # (3,)
    ee_target = ee_pos + x_dot * N * dt    # look-ahead N steps
    q_next, _ = ik.solve(ee_target, ...)
    command q_next as position target
"""

import numpy as np
import pickle
from pathlib import Path


def _log_gauss_pdf(x, mu, sigma):
    d = mu.shape[0]
    diff = x - mu[:, None]
    try:
        L = np.linalg.cholesky(sigma)
        v = np.linalg.solve(L, diff)
        log_det = 2 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        sigma = sigma + (1e-6 - np.linalg.eigvalsh(sigma).min()) * np.eye(d)
        L = np.linalg.cholesky(sigma)
        v = np.linalg.solve(L, diff)
        log_det = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + np.sum(v**2, axis=0))


def posterior_probs(x, priors, mus, sigmas):
    K, N = len(priors), x.shape[1]
    log_apx = np.zeros((K, N))
    for k in range(K):
        log_apx[k] = np.log(priors[k] + 1e-300) + \
                     _log_gauss_pdf(x, mus[:, k], sigmas[:, :, k])
    log_apx -= log_apx.max(axis=0, keepdims=True)
    apx = np.exp(log_apx)
    return apx / apx.sum(axis=0, keepdims=True)


def fit_gmm_bic(x, K_max=10, reg_covar=1e-4):
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        raise ImportError("pip install scikit-learn --break-system-packages")
    best_bic, best_gmm = np.inf, None
    for K in range(1, K_max + 1):
        gmm = GaussianMixture(n_components=K, covariance_type='full',
                              reg_covar=reg_covar, max_iter=500,
                              n_init=5, random_state=42)
        gmm.fit(x)
        bic = gmm.bic(x)
        if bic < best_bic:
            best_bic, best_gmm = bic, gmm
    K = best_gmm.n_components
    print(f"[GMM] BIC selected K={K}  (BIC={best_bic:.1f})")
    return (best_gmm.weights_,
            best_gmm.means_.T,
            best_gmm.covariances_.transpose(1, 2, 0))


def solve_lpv_sdp(x, xdot, x_goal, priors, mus, sigmas,
                  epsilon=1e-3, verbose=False):
    """Fit LPVDS using the same affine convention as the MEAM reference.

    The learned local systems are f_k(x) = A_k x + b_k with the attractor
    constraint b_k = -A_k x_goal.  Keeping b_k explicit mirrors
    book-ds-opt/lpv-opt/optimize_lpv_ds_from_data.m and makes the convention
    less error-prone than folding the attractor into pre-weighted dx terms.
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("pip install cvxpy")
    d, N = x.shape
    K    = len(priors)
    h    = posterior_probs(x, priors, mus, sigmas)

    A_vars = [cp.Variable((d, d)) for _ in range(K)]
    b_vars = [cp.Variable(d) for _ in range(K)]
    constraints = []
    local_vels = []
    for k in range(K):
        constraints += [
            A_vars[k] + A_vars[k].T << -epsilon * np.eye(d),
            b_vars[k] == -A_vars[k] @ x_goal,
        ]
        f_k = A_vars[k] @ x + cp.reshape(b_vars[k], (d, 1), order="F")
        local_vels.append(cp.multiply(h[k, :][None, :], f_k))

    Xdot_hat = sum(local_vels)
    # Match the MEAM/book LPVDS objective more closely: sum per-sample
    # Euclidean velocity errors, rather than a single squared Frobenius loss.
    objective = cp.Minimize(cp.sum(cp.norm(Xdot_hat - xdot, axis=0)))

    prob = cp.Problem(objective, constraints)
    for solver in ['CLARABEL', 'SCS', 'ECOS']:
        try:
            prob.solve(solver=solver, verbose=verbose)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                break
        except Exception:
            continue
    if prob.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f"SDP failed: {prob.status}")
    A_k = np.stack([A_vars[k].value for k in range(K)], axis=-1)
    b_k = np.stack([b_vars[k].value for k in range(K)], axis=-1)
    n_bad = sum(1 for k in range(K)
                if np.any(np.linalg.eigvalsh(A_k[:,:,k] + A_k[:,:,k].T) >= 0))
    if n_bad:
        print(f"[WARN] {n_bad}/{K} A_k not strictly negative definite")
    else:
        print(f"[LPV] All {K} A_k satisfy A+A^T < 0 ✓")
    return A_k, b_k


class LPVDS:
    """3D Cartesian LPVDS.

    predict(ee_pos) -> x_dot (3,).

    Deploy pattern:
        x_dot     = model.predict(ee_pos)
        ee_target = ee_pos + x_dot * lookahead * dt
        q_next, _ = ik.solve(ee_target, ...)
        send q_next as position target
    """

    def __init__(self, priors, mus, sigmas, A_k, b_k, x_goal,
                 x_mean, x_scale):
        self.priors  = priors
        self.mus     = mus       # (3, K) normalised
        self.sigmas  = sigmas    # (3,3,K) normalised
        self.A_k     = A_k      # (3,3,K) normalised
        self.b_k     = b_k      # (3,K) normalised
        self.x_goal  = x_goal   # (3,) raw metres
        self.x_mean  = x_mean   # (3,)
        self.x_scale = x_scale  # (3,)

    def predict(self, x):
        """x: (3,) EE position -> x_dot (3,)."""
        x = np.asarray(x, dtype=float)
        x_n = (x[:3] - self.x_mean) / self.x_scale
        h = posterior_probs(x_n[:, None], self.priors, self.mus, self.sigmas)
        xdot_n = np.zeros(3)
        for k in range(len(self.priors)):
            xdot_n += h[k, 0] * (self.A_k[:, :, k] @ x_n + self.b_k[:, k])
        return xdot_n * self.x_scale

    def safe_velocity(self, x, alpha=0.2):
        """Return a minimally projected velocity that moves toward x_goal.

        The LPVDS SDP is solved in normalised coordinates, but IK deployment
        observes raw Cartesian motion.  This projection is a small runtime
        guard: if the raw velocity would increase squared distance to the
        attractor faster than the requested bound, remove only the offending
        component along the goal error.
        """
        x = np.asarray(x, dtype=float)[:3]
        v = self.predict(x)
        err = x - self.x_goal
        norm_sq = float(np.dot(err, err))
        if norm_sq < 1e-8:
            return v

        dot = float(np.dot(err, v))
        bound = -alpha * norm_sq
        if dot <= bound:
            return v

        return v - ((dot - bound) / norm_sq) * err

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"[LPV] Saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        if "x_scale" not in d or len(np.asarray(d.get("x_mean", []))) != 3:
            raise ValueError(
                f"{path} is not a 3D LPVDS checkpoint. Re-run train_lpvds.py."
            )
        obj = cls.__new__(cls)
        obj.__dict__.update(d)
        return obj

    @classmethod
    def fit(cls, demos, x_goal, K_max=8, epsilon=1e-3, verbose=False):
        max_speed = 1.0   # m/s

        x_list, xdot_list = [], []
        for demo in demos:
            steps = [s for s in demo['trajectory']
                     if s.get('primitive', 'transport') == 'transport']
            for i in range(1, len(steps)):
                ee_prev = np.asarray(steps[i-1]['ee_pos'], dtype=float)[:3]
                ee_curr = np.asarray(steps[i  ]['ee_pos'], dtype=float)[:3]
                dt = float(steps[i].get('physics_dt', 1.0 / 120.0))
                vel     = (ee_curr - ee_prev) / dt
                if np.linalg.norm(vel) > max_speed:
                    continue
                x_list.append(ee_curr)
                xdot_list.append(vel)

        X    = np.stack(x_list,    axis=1)   # (3, N)
        Xdot = np.stack(xdot_list, axis=1)   # (3, N)
        _, N = X.shape
        print(f"[LPV] Fitting 3D Cartesian DS on {N} samples")

        x_mean = X.mean(axis=1)
        raw_std = X.std(axis=1)
        xy_scale = max(float(np.mean(raw_std[:2])), 1e-6)
        z_scale = max(float(raw_std[2]), 0.25 * xy_scale, 0.01)
        x_scale = np.array([xy_scale, xy_scale, z_scale])
        print(f"[LPV] Normalisation scale xyz={np.round(x_scale, 4)} "
              f"(raw std={np.round(raw_std, 4)})")

        X_n    = (X - x_mean[:, None]) / x_scale[:, None]
        x_goal_arr = np.asarray(x_goal, dtype=float)
        x_goal_n   = (x_goal_arr[:3] - x_mean) / x_scale
        Xdot_n     = Xdot / x_scale[:, None]

        priors, mus_n, sigmas_n = fit_gmm_bic(X_n.T, K_max=K_max)
        print("[LPV] Solving SDP...")
        A_k, b_k = solve_lpv_sdp(X_n, Xdot_n, x_goal_n,
                                  priors, mus_n, sigmas_n,
                                  epsilon=epsilon, verbose=verbose)

        model = cls(priors=priors, mus=mus_n, sigmas=sigmas_n,
                    A_k=A_k, b_k=b_k, x_goal=x_goal_arr,
                    x_mean=x_mean, x_scale=x_scale)
        dots = []
        for i in range(X.shape[1]):
            x_raw = X[:, i]
            dots.append(float(np.dot(x_raw - x_goal_arr, model.predict(x_raw))))
        dots = np.asarray(dots)
        frac_inward = float(np.mean(dots < 0.0))
        print(f"[LPV] Inward velocity fraction on demos: {frac_inward*100:.1f}% "
              f"(median dot={np.median(dots):.5f})")
        return model


def evaluate_lpvds(model, demos):
    errs = []
    for demo in demos:
        steps = [s for s in demo['trajectory']
                 if s.get('primitive', 'transport') == 'transport']
        for i in range(1, len(steps)):
            ee_prev = np.asarray(steps[i-1]['ee_pos'])[:3]
            ee_curr = np.asarray(steps[i  ]['ee_pos'])[:3]
            dt = float(steps[i].get('physics_dt', 1.0 / 120.0))
            xdot_true = (ee_curr - ee_prev) / dt
            if np.linalg.norm(xdot_true) > 1.0:
                continue
            pred = model.predict(np.asarray(steps[i]['ee_pos']))
            errs.append(np.linalg.norm(pred - xdot_true))
    rmse = np.sqrt(np.mean(np.array(errs)**2))
    print(f"[LPV] Cartesian velocity RMSE: {rmse*1000:.2f} mm/s")
    return rmse
