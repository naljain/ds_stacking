"""
Joint-space Neural Dynamical System with learned Lyapunov function.

State:    q ∈ R^7   (Franka arm joints, fingers excluded)
Goal:     q* ∈ R^7  (target joint configuration for the primitive)
Velocity: q̇ = f_theta(q, q*) ∈ R^7

Lyapunov candidate (positive definite around q*, by construction):
    V(q, q*) = ||g(q) - g(q*)||² + epsilon * ||q - q*||²

Stability training enforces dV/dt = ∇_q V · f(q, q*) < -alpha · V on the data
distribution. At inference time we additionally provide a Lyapunov projection
that gives EXACT stability per-step rather than soft-trained stability.

This module replaces the EE-space DS used in the previous iteration. The
deployment pipeline now does NOT integrate velocity then IK — it commands
joint velocities directly to the articulation, so the closed-loop dynamics
are exactly the trained DS modulo physical actuation.

Inputs to forward() are the concatenation [q, q*] ∈ R^14, which keeps the
goal as part of the state (DS is "goal-conditioned"). Internally we slice
the goal back out for the Lyapunov computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


N_JOINTS = 7  # Franka arm joints (fingers handled separately by gripper)
STATE_DIM = 2 * N_JOINTS  # [q (7), q_goal (7)]


class NeuralDS(nn.Module):
    """Joint-velocity field f_theta(q, q*) -> R^7."""

    def __init__(self, state_dim=STATE_DIM, hidden_dim=128, n_joints=N_JOINTS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_joints),
        )

    def forward(self, x):
        return self.net(x)


class LyapunovNet(nn.Module):
    """V_phi(q, q*) — positive definite around q*.

    V(q, q*) = ||g([q, q*]) - g([q*, q*])||² + epsilon * ||q - q*||²

    The first term goes to 0 when q = q* by construction; the second term
    guarantees positive-definiteness so V is a valid Lyapunov candidate.
    """

    def __init__(self, n_joints=N_JOINTS, hidden_dim=64, epsilon=0.01):
        super().__init__()
        self.n_joints = n_joints
        self.epsilon  = epsilon
        self.g = nn.Sequential(
            nn.Linear(2 * n_joints, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # x = [q (7), q_goal (7)]
        q       = x[..., :self.n_joints]
        q_goal  = x[..., self.n_joints:]

        # Reference state with q replaced by q_goal — the "x at goal"
        x_at_goal = torch.cat([q_goal, q_goal], dim=-1)

        psd = ((self.g(x) - self.g(x_at_goal)) ** 2).sum(dim=-1)
        reg = self.epsilon * ((q - q_goal) ** 2).sum(dim=-1)
        return psd + reg


class StableNeuralDS(nn.Module):
    """Joint-space DS with Lyapunov stability machinery."""

    def __init__(self, n_joints=N_JOINTS, hidden_dim=128, lyap_hidden=64,
                 alpha=1.0):
        super().__init__()
        self.n_joints = n_joints
        self.f = NeuralDS(state_dim=2 * n_joints, hidden_dim=hidden_dim,
                          n_joints=n_joints)
        self.V = LyapunovNet(n_joints=n_joints, hidden_dim=lyap_hidden)
        self.alpha = alpha

    def forward(self, x):
        return self.f(x)

    def lyapunov(self, x):
        return self.V(x)

    def safe_velocity(self, x):
        """Return f(x) projected onto {v : ∇_q V · v <= -alpha · V}.

        We compute ∇_q V (only the q part, not q_goal) and project the
        velocity if the closed-loop derivative would be too large. This
        guarantees per-step decrease of V regardless of f's training quality.
        """
        with torch.enable_grad():
            x_g = x.detach().clone().requires_grad_(True)
            V_val = self.V(x_g)
            grad = torch.autograd.grad(V_val.sum(), x_g)[0]
        gV_q = grad[..., :self.n_joints]   # only q-part matters for q̇

        with torch.no_grad():
            v_raw = self.f(x)
            dot   = (gV_q * v_raw).sum(dim=-1, keepdim=True)
            bound = -self.alpha * V_val.unsqueeze(-1)

            norm_sq = (gV_q ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            excess  = (dot - bound).clamp(min=0.0)
            v_safe  = v_raw - (excess / norm_sq) * gV_q
        return v_safe


# ── Loss functions ───────────────────────────────────────────────────────────
def imitation_loss(model, x, q_dot_demo):
    return F.mse_loss(model(x), q_dot_demo)


def stability_loss(model, x, alpha=1.0):
    """Enforce dV/dt + alpha · V <= 0 on the training distribution."""
    x_g = x.detach().clone().requires_grad_(True)
    V_val = model.V(x_g)
    grad  = torch.autograd.grad(V_val.sum(), x_g, create_graph=True)[0]
    gV_q  = grad[..., :model.n_joints]

    v     = model.f(x_g)
    dV_dt = (gV_q * v).sum(dim=-1)

    return F.relu(dV_dt + alpha * V_val).mean()


def total_loss(model, x, q_dot, alpha=1.0, lambda_stab=0.5):
    L_imit = imitation_loss(model, x, q_dot)
    L_stab = stability_loss(model, x, alpha)
    return L_imit + lambda_stab * L_stab, L_imit.item(), L_stab.item()
