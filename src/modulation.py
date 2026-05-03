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
                 tail_effect=True, eta_min=0.05):
        self.safe_radius = safe_radius
        self.reactivity  = reactivity
        self.tail_effect = tail_effect
        self.eta_min     = eta_min

    # ── Level-set Γ and reference direction ────────────────────────────────
    def gamma(self, x, x_obs):
        d = np.linalg.norm(x - x_obs)
        return (d / self.safe_radius) ** self.reactivity

    def reference_direction(self, x, x_obs):
        diff = x - x_obs
        return diff / (np.linalg.norm(diff) + 1e-9)

    # ── Construct E(x), the basis aligned with reference direction ─────────
    def _build_basis(self, r):
        """Return a 3x3 orthonormal matrix whose first column is r."""
        r = r / (np.linalg.norm(r) + 1e-12)
        # Match the MEAM HW 3D construction: normal plus two tangents.  Choose
        # a helper axis that is not parallel to r, then Gram-Schmidt.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, r)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        t1 = helper - np.dot(helper, r) * r
        t1 = t1 / (np.linalg.norm(t1) + 1e-12)
        t2 = np.cross(r, t1)
        t2 = t2 / (np.linalg.norm(t2) + 1e-12)
        return np.column_stack([r, t1, t2])

    # ── Modulation matrix M(x) = E D E^T ────────────────────────────────────
    def modulation_matrix(self, x, x_obs, v_nom=None):
        """Construct M(x) for current state x and obstacle x_obs.

        If v_nom is provided and tail_effect is enabled, suppress radial
        damping when the velocity already points outward.
        """
        Γ = self.gamma(x, x_obs)
        # Far away -> no modulation. Keep this threshold generous so a larger
        # EE safety sphere creates an early, smooth avoidance field.
        if Γ > 100.0:
            return np.eye(3)

        r = self.reference_direction(x, x_obs)
        E = self._build_basis(r)

        # Default Huber eigenvalues
        if Γ >= 1.0:
            λ_r = 1.0 - 1.0 / Γ
            λ_t = 1.0 + 1.0 / Γ
        else:
            # Inside the safety sphere — push out, but clamp for numerics
            λ_r = self.eta_min
            λ_t = 2.0  # full tangential boost

        # Tail effect: if v_nom is already heading outward (positive component
        # along r), no need to damp the radial component
        if self.tail_effect and v_nom is not None:
            if np.dot(v_nom, r) >= 0.0:
                λ_r = 1.0

        D = np.diag([λ_r, λ_t, λ_t])
        return E @ D @ E.T

    # ── Apply modulation to a Cartesian velocity ───────────────────────────
    def modulate_cartesian(self, v_cart, ee_self, ee_other):
        M = self.modulation_matrix(ee_self, ee_other, v_nom=v_cart)
        return M @ v_cart


# ────────────────────────────────────────────────────────────────────────────
#  Inter-arm wrapper that applies modulation in joint space
# ────────────────────────────────────────────────────────────────────────────
class InterArmModulation:
    """Compose Huber modulation with a Jacobian projection to act on q̇."""

    def __init__(self, safe_radius=0.15, reactivity=2.0,
                 tail_effect=True, eta_min=0.05, jac_damping=0.05):
        self.huber = HuberModulation(
            safe_radius=safe_radius,
            reactivity=reactivity,
            tail_effect=tail_effect,
            eta_min=eta_min,
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
        return {
            "gamma":           self.huber.gamma(ee_pos_self, ee_pos_other),
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
