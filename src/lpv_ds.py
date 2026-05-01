"""
LPV-DS in Cartesian (EE position) space.

The DS learns a stable vector field in R^3:
    x_dot = Σ_k h_k(x) * A_k * (x - x_goal)

where:
  x       : EE position (3,)
  x_goal  : fixed pre-place EE position above the stack (3,)
  h_k(x)  : GMM posterior probabilities (soft assignment to K local models)
  A_k     : 3×3 stable matrices (A_k + A_k^T ≺ 0, via SDP)

At deploy time the Cartesian velocity x_dot_des is converted to joint
velocities via the translational Jacobian pseudoinverse:
    q_dot = J^+ @ x_dot_des        J = jacobian_finite_difference(franka)

Why Cartesian?
  Joint-space trajectories for the same EE motion look very different
  across blocks (IK non-uniqueness scrambles the joint configuration),
  so the GMM cannot find meaningful clusters.  In Cartesian space all
  transport trajectories start at scattered (x,y,z_lift) positions and
  converge to the same x_goal — a clean flow that LPVDS was designed for.

Reference:
  "A Physically-Consistent Bayesian Non-Parametric Mixture Model for
   Dynamical System Learning"; N. Figueroa and A. Billard; CoRL 2018
"""

import numpy as np
import pickle
from pathlib import Path


# ── Gaussian mixture posterior ────────────────────────────────────────────────

def _log_gauss_pdf(x, mu, sigma):
    """Log of multivariate Gaussian pdf.  x: (d,N), mu: (d,), sigma: (d,d)."""
    d = mu.shape[0]
    diff = x - mu[:, None]
    try:
        L = np.linalg.cholesky(sigma)
        v = np.linalg.solve(L, diff)
        log_det = 2 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(sigma)
        sigma = sigma + (1e-6 - eigvals.min()) * np.eye(d)
        L = np.linalg.cholesky(sigma)
        v = np.linalg.solve(L, diff)
        log_det = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + np.sum(v**2, axis=0))


def posterior_probs(x, priors, mus, sigmas):
    """Normalised GMM posteriors h_k(x).
    x: (d,N), returns (K,N)."""
    K, N = len(priors), x.shape[1]
    log_apx = np.zeros((K, N))
    for k in range(K):
        log_apx[k] = np.log(priors[k] + 1e-300) + \
                     _log_gauss_pdf(x, mus[:, k], sigmas[:, :, k])
    log_apx -= log_apx.max(axis=0, keepdims=True)
    apx = np.exp(log_apx)
    return apx / apx.sum(axis=0, keepdims=True)


# ── GMM fitting ───────────────────────────────────────────────────────────────

def fit_gmm_bic(x, K_max=10, reg_covar=1e-4):
    """Fit full-covariance GMM with BIC model selection.
    x: (N, d). Returns priors (K,), mus (d,K), sigmas (d,d,K)."""
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
    priors = best_gmm.weights_                         # (K,)
    mus    = best_gmm.means_.T                         # (d, K)
    sigmas = best_gmm.covariances_.transpose(1, 2, 0)  # (d, d, K)
    return priors, mus, sigmas


# ── SDP ───────────────────────────────────────────────────────────────────────

def solve_lpv_sdp(x, xdot, x_goal, priors, mus, sigmas,
                  epsilon=1e-3, verbose=False):
    """Solve for stable {A_k} in Cartesian space.

    min  || Σ_k h_k(x_n) A_k (x_n - x_goal) - xdot_n ||_F^2
    s.t. A_k + A_k^T ≼ -ε I   ∀k

    Args:
        x, xdot : (3, N)
        x_goal  : (3,)
        priors, mus, sigmas : GMM params in normalised space
    Returns:
        A_k (3,3,K), b_k (3,K)  where b_k = -A_k @ x_goal
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("pip install cvxpy")

    d, N = x.shape
    K    = len(priors)

    h   = posterior_probs(x, priors, mus, sigmas)   # (K, N)
    dx  = x - x_goal[:, None]                       # (d, N)

    A_vars = [cp.Variable((d, d)) for _ in range(K)]

    # Vectorised: Dxk[:,k] = dx weighted by h[k,:]
    Dxk      = [dx * h[k, :][None, :] for k in range(K)]
    Xdot_hat = sum(A_vars[k] @ Dxk[k] for k in range(K))   # (d, N)
    Error    = Xdot_hat - xdot

    objective   = cp.Minimize(cp.sum_squares(Error))
    constraints = [A_vars[k] + A_vars[k].T << -epsilon * np.eye(d)
                   for k in range(K)]

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

    A_k = np.stack([A_vars[k].value for k in range(K)], axis=-1)   # (d,d,K)
    b_k = np.stack([-A_vars[k].value @ x_goal for k in range(K)],
                   axis=-1)                                          # (d,K)

    n_bad = sum(
        1 for k in range(K)
        if np.any(np.linalg.eigvalsh(A_k[:,:,k] + A_k[:,:,k].T) >= 0)
    )
    if n_bad:
        print(f"[WARN] {n_bad}/{K} A_k not strictly negative definite")
    else:
        print(f"[LPV] All {K} A_k satisfy A+A^T < 0 ✓")

    return A_k, b_k


# ── LPVDS class ───────────────────────────────────────────────────────────────

class LPVDS:
    """2D Cartesian-space LPV-DS operating in the XY plane.

    Z is nearly constant during transport (lift height), so operating in 2D
    avoids the DS being dominated by tiny Z variations after normalisation.

    predict(x) takes a 3D EE position, strips Z, evaluates the 2D DS,
    and returns a 3D velocity with x_dot[2]=0.
    At deploy: q_dot = ik.solve(ee_pos + x_dot*dt) as IK velocity.
    """

    def __init__(self, priors, mus, sigmas, A_k, b_k, x_goal,
                 x_mean, x_std):
        self.priors  = priors    # (K,)
        self.mus     = mus       # (3, K)  — in normalised space
        self.sigmas  = sigmas    # (3,3,K) — in normalised space
        self.A_k     = A_k      # (3,3,K) — in normalised space
        self.b_k     = b_k      # (3,K)   — in normalised space
        self.x_goal  = x_goal   # (3,)    — raw metres
        self.x_mean  = x_mean   # (3,)    position centroid
        self.x_std   = x_std    # scalar  isotropic scale

    def predict(self, x):
        """Evaluate 2D DS at EE position x.

        Args:
            x : (3,) or (2,) EE position — only XY is used
        Returns:
            xdot : (3,) with xdot[2]=0, or (2,) if input was 2D
        """
        x = np.asarray(x, dtype=float)
        input_3d = x.shape[0] == 3
        xy = x[:2]                                         # (2,)

        # Normalise XY
        xy_n = (xy - self.x_mean) / self.x_std            # (2,)
        xy_n = xy_n[:, None]                               # (2,1)

        h = posterior_probs(xy_n, self.priors, self.mus, self.sigmas)  # (K,1)

        xydot_n = np.zeros(2)
        for k in range(len(self.priors)):
            xydot_n += h[k, 0] * (self.A_k[:, :, k] @ xy_n[:, 0]
                                   + self.b_k[:, k])

        xydot = xydot_n * self.x_std                       # (2,) in m/s

        if input_3d:
            return np.array([xydot[0], xydot[1], 0.0])    # (3,) z=0
        return xydot

    def safe_velocity(self, x):
        """Stability guaranteed by SDP — shim that just calls predict()."""
        return self.predict(x)

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
        obj = cls.__new__(cls)
        obj.__dict__.update(d)
        return obj

    @classmethod
    def fit(cls, demos, x_goal, K_max=8, epsilon=1e-3, verbose=False):
        """Fit Cartesian LPVDS from collected demos.

        Args:
            demos   : list of demo dicts (from collect_ik.py)
            x_goal  : (3,) EE position at transport target (world metres)
            K_max   : max GMM components (BIC selects best K ≤ K_max)
            epsilon : negative-definiteness margin
        """
        # ── 1. Stack EE XY positions and velocities (2D) ────────────────────
        # Z is nearly constant during transport (lift height), so we operate
        # in 2D XY only. This prevents tiny Z variations from dominating
        # the normalised space and distorting the learned velocity directions.
        dt        = demos[0]['trajectory'][0].get('physics_dt', 1.0 / 120.0)                     if demos else 1.0 / 120.0
        max_speed = 1.0   # m/s XY speed — faster is a sim artifact

        x_list, xdot_list = [], []
        for demo in demos:
            steps = [s for s in demo['trajectory']
                     if s.get('primitive', 'transport') == 'transport']
            for i in range(1, len(steps)):
                ee_prev = np.asarray(steps[i-1]['ee_pos'])[:2]   # XY only
                ee_curr = np.asarray(steps[i  ]['ee_pos'])[:2]
                vel     = (ee_curr - ee_prev) / dt
                if np.linalg.norm(vel) > max_speed:
                    continue
                x_list.append(ee_curr)
                xdot_list.append(vel)

        X    = np.stack(x_list,    axis=1)    # (2, N)
        Xdot = np.stack(xdot_list, axis=1)   # (2, N)
        _, N = X.shape
        print(f"[LPV] Fitting 2D (XY) DS on {N} samples")

        # ── 2. Isotropic normalisation ────────────────────────────────────────
        # Use a single scalar std (not per-dimension) so velocity directions
        # are preserved. Per-dimension scaling distorts directions and causes
        # near-zero std on fixed axes (e.g. z is constant during transport).
        x_mean = X.mean(axis=1)                          # (3,) centroid
        x_std  = float(X.std())                          # scalar global scale
        x_std  = max(x_std, 1e-6)
        X_n    = (X - x_mean[:, None]) / x_std           # (3, N)

        x_goal_arr = np.asarray(x_goal, dtype=float)[:2]  # XY only
        x_goal_n   = (x_goal_arr - x_mean) / x_std       # (2,)

        # Normalise velocities with the same scalar — direction preserved
        Xdot_n = Xdot / x_std

        # ── 3. Fit GMM in normalised EE space ─────────────────────────────────
        priors, mus_n, sigmas_n = fit_gmm_bic(X_n.T, K_max=K_max)

        # ── 4. Solve SDP ──────────────────────────────────────────────────────
        print("[LPV] Solving SDP…")
        A_k, b_k = solve_lpv_sdp(X_n, Xdot_n, x_goal_n,
                                  priors, mus_n, sigmas_n,
                                  epsilon=epsilon, verbose=verbose)

        x_goal_3d = np.asarray(x_goal, dtype=float)   # full 3D for convergence
        return cls(priors=priors, mus=mus_n, sigmas=sigmas_n,
                   A_k=A_k, b_k=b_k, x_goal=x_goal_3d,
                   x_mean=x_mean, x_std=x_std)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_lpvds(model, demos):
    errs = []
    for demo in demos:
        steps = [s for s in demo['trajectory']
                 if s.get('primitive', 'transport') == 'transport']
        for i, step in enumerate(steps):
            if 'ee_vel' not in step and i == 0:
                continue
            x    = np.asarray(step['ee_pos'])
            xdot = np.asarray(step.get('ee_vel', np.zeros(3)))
            pred = model.predict(x)
            errs.append(np.linalg.norm(pred - xdot))
    rmse = np.sqrt(np.mean(np.array(errs)**2))
    print(f"[LPV] EE velocity RMSE: {rmse*1000:.2f} mm/s")
    return rmse
