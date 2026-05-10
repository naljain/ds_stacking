"""
Dynamical-system modulation for inter-arm collision avoidance.

Implements the modulation framework of:
  Huber, Billard, Slotine (2019) — "Avoidance of Convex and Concave Obstacles
  with Convergence Ensured Through Contraction." IEEE RA-L.

This extends the original Khansari-Zadeh & Billard formulation by:
  (a) Using a *reference point* R inside the obstacle to define the direction
      of deflection, rather than the surface normal. This avoids the local-
      minimum / saddle-point issue at the obstacle's antipode.
  (b) Adding a *tail-effect* gating so that velocities already pointing away
      from the obstacle are not modulated at all. This recovers nominal motion
      whenever it is safe to do so.
  (c) Providing a contraction-based convergence guarantee for the closed-loop
      modulated DS to its attractor, even with the obstacle present.

We treat the other arm's EE as a moving spherical obstacle. The reference
point is placed at the obstacle centre (i.e. at the other arm's EE).

────────────────────────────────────────────────────────────────────────────
Construction
────────────────────────────────────────────────────────────────────────────

For a state x and obstacle centre x_obs (radius R):

    Γ(x) = (||x - x_obs|| / R)^p          obstacle level-set function
                                          Γ > 1 outside, Γ = 1 on boundary

    r(x) = (x - x_obs) / ||x - x_obs||    reference direction (unit vector
                                          pointing from R out through x)

The modulation matrix is

    M(x) = E(x) · D(x) · E(x)^{-1}

where:
  - E(x) is a basis whose FIRST column is r(x) (the reference direction),
    and the remaining columns span the orthogonal complement.
  - D(x) = diag(λ_r, λ_t, λ_t, ...), the eigenvalues are
        λ_r(x) = 1 - 1/Γ(x)         (radial — squashed near surface)
        λ_t(x) = 1 + 1/Γ(x)         (tangential — boosted near surface)
    with the tail-effect rule: if the nominal velocity already has positive
    component along r (already moving away), set λ_r := 1 (no radial damping).

The full modulated velocity is

    v_mod(x) = M(x) · v_nom(x)

For multiple obstacles, Huber 2019 proposes weighted blending; we have only
one (the other arm) so we use the single-obstacle form.

────────────────────────────────────────────────────────────────────────────
Joint-space wrapping
────────────────────────────────────────────────────────────────────────────

Our DS produces joint velocities q̇. We:
  1. Map to Cartesian: v_cart = J_trans(q) · q̇
  2. Modulate:         v_cart_mod = M(ee) · v_cart
  3. Map back:         q̇_mod = q̇ + J_trans^+ · (v_cart_mod - v_cart)

The +Δ form preserves joint motions in the null-space of J_trans (so e.g.
elbow re-positioning is not disturbed by the modulation).
"""

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
#  Core Huber-2019 modulation
# ────────────────────────────────────────────────────────────────────────────
class HuberModulation:
    """Single-obstacle Huber 2019 modulation in 3D Cartesian space.

    Args:
        safe_radius : sphere radius around the other arm's EE (metres)
        reactivity  : exponent p in Γ. Higher → modulation only in close range.
        tail_effect : if True (default), don't modulate when velocity already
                      points outward (Huber 2019 §III-C). Crucial for letting
                      the arm leave the contention zone naturally.
        eta_min     : lower bound on radial eigenvalue λ_r when *inside* the
                      safety sphere (Γ ≤ 1). Strict Huber would allow λ_r ≤ 0
                      (push out hard) but in practice we clamp to a small
                      positive value to avoid joint-velocity spikes through
                      the Jacobian-pseudoinverse.
    """

    def __init__(self, safe_radius=0.15, reactivity=2.0,
                 tail_effect=True, eta_min=0.05, preserve_speed=True,
                 isoline=1.0, max_pairs=4):
        self.safe_radius = safe_radius
        self.reactivity  = reactivity
        self.tail_effect = tail_effect
        self.eta_min     = eta_min
        self.preserve_speed = preserve_speed
        self.isoline = max(float(isoline), 1e-6)
        self.max_pairs = None if max_pairs is None else max(1, int(max_pairs))

    # ── Level-set Γ and reference direction ────────────────────────────────
    def gamma(self, x, x_obs):
        d = np.linalg.norm(x - x_obs)
        return (d / self.safe_radius) ** self.reactivity

    def gamma_eff(self, x, x_obs):
        return self.gamma(x, x_obs) / self.isoline

    def reference_direction(self, x, x_obs):
        diff = x - x_obs
        return diff / (np.linalg.norm(diff) + 1e-9)

    # ── Construct E(x), the basis aligned with reference direction ─────────
    def _build_basis(self, r):
        """Return a 3x3 orthonormal matrix whose first column is r."""
        r = r / (np.linalg.norm(r) + 1e-12)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, r)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        t1 = helper - np.dot(helper, r) * r
        t1 = t1 / (np.linalg.norm(t1) + 1e-12)
        t2 = np.cross(r, t1)
        t2 = t2 / (np.linalg.norm(t2) + 1e-12)
        return np.column_stack([r, t1, t2])

    def _eigenvalues(self, gamma, r, v_nom=None):
        if gamma > 1.0:
            lambda_r = 1.0 - 1.0 / gamma
            lambda_t = 1.0 + 1.0 / gamma
        else:
            lambda_r = self.eta_min
            lambda_t = 2.0

        tail_active = False
        if self.tail_effect and v_nom is not None and np.dot(v_nom, r) >= 0.0:
            lambda_r = 1.0
            tail_active = True

        return lambda_r, lambda_t, tail_active

    def components(self, x, x_obs, v_nom=None):
        gamma = self.gamma(x, x_obs)
        gamma_eff = gamma / self.isoline
        r = self.reference_direction(x, x_obs)
        lambda_r, lambda_t, tail_active = self._eigenvalues(gamma_eff, r, v_nom)
        return {
            "gamma": float(gamma),
            "gamma_eff": float(gamma_eff),
            "isoline": float(self.isoline),
            "reference_direction": r,
            "lambda_r": float(lambda_r),
            "lambda_t": float(lambda_t),
            "tail_active": bool(tail_active),
            "preserve_speed": bool(self.preserve_speed),
        }

    # ── Modulation matrix M(x) = E D E^T ────────────────────────────────────
    def modulation_matrix(self, x, x_obs, v_nom=None):
        """Construct M(x) for current state x and obstacle x_obs.

        If v_nom is provided and tail_effect is enabled, suppress radial
        damping when the velocity already points outward.
        """
        Γ = self.gamma_eff(x, x_obs)
        # Far away → no modulation
        if Γ > 100.0:
            return np.eye(3)

        r = self.reference_direction(x, x_obs)
        E = self._build_basis(r)
        λ_r, λ_t, _ = self._eigenvalues(Γ, r, v_nom)

        D = np.diag([λ_r, λ_t, λ_t])
        return E @ D @ E.T

    # ── Apply modulation to a Cartesian velocity ───────────────────────────
    def modulate_cartesian(self, v_cart, ee_self, ee_other):
        M = self.modulation_matrix(ee_self, ee_other, v_nom=v_cart)
        v_mod = M @ v_cart
        if self.preserve_speed:
            nom_norm = np.linalg.norm(v_cart)
            mod_norm = np.linalg.norm(v_mod)
            if nom_norm > 1e-9 and mod_norm > 1e-9:
                v_mod = v_mod * (nom_norm / mod_norm)
        return v_mod

    def modulate_cartesian_points(self, v_cart, self_points, obstacle_points,
                                  max_pairs=None):
        """Apply sequential modulation from the closest protected point pairs."""
        self_points = np.asarray(self_points, dtype=float).reshape(-1, 3)
        obstacle_points = np.asarray(obstacle_points, dtype=float).reshape(-1, 3)
        if len(self_points) == 0 or len(obstacle_points) == 0:
            return v_cart

        pairs = []
        for i, p_self in enumerate(self_points):
            for j, p_obs in enumerate(obstacle_points):
                pairs.append((self.gamma_eff(p_self, p_obs), i, j))
        pairs.sort(key=lambda item: item[0])

        pair_limit = self.max_pairs if max_pairs is None else max_pairs
        active_pairs = pairs if pair_limit is None else pairs[:max(1, int(pair_limit))]
        v_mod = np.asarray(v_cart, dtype=float).copy()
        for _, i, j in active_pairs:
            v_mod = self.modulate_cartesian(v_mod, self_points[i], obstacle_points[j])
        return v_mod

    def closest_components(self, self_points, obstacle_points, v_nom=None):
        """Diagnostics for the closest protected sphere pair."""
        self_points = np.asarray(self_points, dtype=float).reshape(-1, 3)
        obstacle_points = np.asarray(obstacle_points, dtype=float).reshape(-1, 3)
        best = None
        for i, p_self in enumerate(self_points):
            for j, p_obs in enumerate(obstacle_points):
                gamma_eff = self.gamma_eff(p_self, p_obs)
                if best is None or gamma_eff < best[0]:
                    best = (gamma_eff, i, j, p_self, p_obs)
        if best is None:
            return None
        _, i, j, p_self, p_obs = best
        comp = self.components(p_self, p_obs, v_nom=v_nom)
        comp.update({
            "self_point_index": int(i),
            "obstacle_point_index": int(j),
            "self_point": p_self,
            "obstacle_point": p_obs,
            "distance": float(np.linalg.norm(p_self - p_obs)),
        })
        return comp


# ────────────────────────────────────────────────────────────────────────────
#  Inter-arm wrapper that applies modulation in joint space
# ────────────────────────────────────────────────────────────────────────────
class InterArmModulation:
    """Compose Huber modulation with a Jacobian projection to act on q̇."""

    def __init__(self, safe_radius=0.15, reactivity=2.0,
                 tail_effect=True, eta_min=0.05, jac_damping=0.05,
                 preserve_speed=True, isoline=1.0, max_pairs=4):
        self.huber = HuberModulation(
            safe_radius=safe_radius,
            reactivity=reactivity,
            tail_effect=tail_effect,
            eta_min=eta_min,
            preserve_speed=preserve_speed,
            isoline=isoline,
            max_pairs=max_pairs,
        )
        self.jac_damping = jac_damping

    def modulate_joint_velocity(self, q_dot_nominal,
                                ee_pos_self, ee_pos_other,
                                jacobian):
        """Modulate q̇ via Cartesian Huber modulation + damped Jacobian inverse.

        v_cart_nom = J_trans · q̇_nom
        v_cart_mod = M(ee, ee_other) · v_cart_nom            (Huber)
        Δv         = v_cart_mod - v_cart_nom
        q̇_mod      = q̇_nom + J_trans^+ · Δv                  (only Cartesian
                                                              correction is
                                                              added back)
        """
        J_trans = jacobian[:3, :]
        v_nom   = J_trans @ q_dot_nominal
        v_mod   = self.huber.modulate_cartesian(v_nom, ee_pos_self, ee_pos_other)
        Δv      = v_mod - v_nom

        # Damped least-squares pseudoinverse
        JJt    = J_trans @ J_trans.T
        damp   = (self.jac_damping ** 2) * np.eye(JJt.shape[0])
        J_pinv = J_trans.T @ np.linalg.inv(JJt + damp)

        return q_dot_nominal + J_pinv @ Δv

    def modulate_joint_velocity_points(self, q_dot_nominal,
                                       self_points, obstacle_points,
                                       jacobian):
        """Joint-space wrapper for EE plus wrist/gripper proxy spheres."""
        J_trans = jacobian[:3, :]
        v_nom = J_trans @ q_dot_nominal
        v_mod = self.huber.modulate_cartesian_points(
            v_nom, self_points, obstacle_points
        )
        delta_v = v_mod - v_nom

        JJt = J_trans @ J_trans.T
        damp = (self.jac_damping ** 2) * np.eye(JJt.shape[0])
        J_pinv = J_trans.T @ np.linalg.inv(JJt + damp)

        return q_dot_nominal + J_pinv @ delta_v

    # ── Diagnostics: useful for the writeup ────────────────────────────────
    def diagnostics(self, q_dot_nominal, ee_pos_self, ee_pos_other, jacobian):
        """Return scalar quantities for plotting/logging:
            gamma            obstacle level-set value
            v_cart_norm_nom  ||v_cart_nominal||
            v_cart_norm_mod  ||v_cart_modulated||
            radial_dot_nom   v_cart_nominal · r        (positive => outward)
            radial_dot_mod   v_cart_modulated · r
        """
        J_trans = jacobian[:3, :]
        v_nom   = J_trans @ q_dot_nominal
        v_mod   = self.huber.modulate_cartesian(v_nom, ee_pos_self, ee_pos_other)
        r       = self.huber.reference_direction(ee_pos_self, ee_pos_other)
        comp = self.huber.components(ee_pos_self, ee_pos_other, v_nom=v_nom)
        return {
            "gamma":           comp["gamma"],
            "gamma_eff":       comp["gamma_eff"],
            "isoline":         comp["isoline"],
            "lambda_r":        comp["lambda_r"],
            "lambda_t":        comp["lambda_t"],
            "tail_active":     comp["tail_active"],
            "v_cart_norm_nom": float(np.linalg.norm(v_nom)),
            "v_cart_norm_mod": float(np.linalg.norm(v_mod)),
            "radial_dot_nom":  float(np.dot(v_nom, r)),
            "radial_dot_mod":  float(np.dot(v_mod, r)),
        }


# ────────────────────────────────────────────────────────────────────────────
#  Jacobian helper (finite-difference fallback)
# ────────────────────────────────────────────────────────────────────────────
def jacobian_finite_difference(franka, eps=1e-4):
    """Translation Jacobian (top 3 rows of a 6×7 Jacobian) via forward
    differences on joint positions. Slow but version-portable. Replace with
    Isaac Sim's analytical Jacobian once the API stabilises in your install.
    """
    q0 = franka.get_joint_positions()[:7].copy()
    ee0, _ = franka.end_effector.get_world_pose()
    ee0 = ee0.copy()

    n = 7
    J = np.zeros((6, n))
    full = franka.get_joint_positions().copy()
    for i in range(n):
        q_pert = q0.copy()
        q_pert[i] += eps
        full[:7] = q_pert
        franka.set_joint_positions(full)
        ee_p, _ = franka.end_effector.get_world_pose()

        # Restore
        full[:7] = q0
        franka.set_joint_positions(full)

        J[:3, i] = (ee_p - ee0) / eps
    return J
