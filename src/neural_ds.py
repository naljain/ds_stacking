"""
Joint-space Neural Dynamical System with learned Lyapunov function.

State:    e = q - q* ∈ R^7   (joint error relative to current goal)
Velocity: q̇ = f_theta(e) ∈ R^7

Using the error e = q - q* as input rather than [q, q*] ∈ R^14 eliminates
the null-space distribution mismatch: the error distribution (large at
primitive start, zero at goal) is the same regardless of which IK solver
computed q*. The [q, q*] formulation caused q* to appear OOD at deployment
because RMPflow and Lula IK settle to different null-space configurations,
making state_std for q* dimensions near zero during training.

Lyapunov candidate (positive definite around e=0 by construction):
    V(e) = ||g(e) - g(0)||² + epsilon * ||e||²

Stability: dV/dt = ∇_e V · ė = ∇_e V · q̇ ≤ -alpha · V on data distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


N_JOINTS = 7  # Franka arm joints (fingers handled separately by gripper)
STATE_DIM = N_JOINTS      # e = q - q_goal (7)


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
        # f(x) = net(x) - net(0) guarantees f(0) = 0 (hard equilibrium at the
        # goal). No -x skip: it gets cancelled during training (net learns
        # net(x) ≈ x + residual), which was hurting OOD behaviour. Convergence
        # is now provided by the Lyapunov projection at deployment (--use_safe).
        zero = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
        return self.net(x) - self.net(zero)


class LyapunovNet(nn.Module):
    """V_phi(e) — positive definite around e = 0.

    V(e) = ||g(e) - g(0)||² + epsilon * ||e||²

    Both terms are zero at e=0 and positive elsewhere.
    Input e = q - q_goal so the "at goal" state is the zero vector.
    """

    def __init__(self, n_joints=N_JOINTS, hidden_dim=64, epsilon=0.5):
        super().__init__()
        self.n_joints = n_joints
        self.epsilon  = epsilon
        self.g = nn.Sequential(
            nn.Linear(n_joints, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # x = e = q - q_goal, shape (..., 7)
        x_at_goal = torch.zeros_like(x)
        psd = ((self.g(x) - self.g(x_at_goal)) ** 2).sum(dim=-1)
        reg = self.epsilon * (x ** 2).sum(dim=-1)
        return psd + reg


class StableNeuralDS(nn.Module):
    """Joint-space DS with Lyapunov stability machinery."""

    def __init__(self, n_joints=N_JOINTS, hidden_dim=128, lyap_hidden=64,
                 alpha=1.0):
        super().__init__()
        self.n_joints = n_joints
        self.f = NeuralDS(state_dim=n_joints, hidden_dim=hidden_dim,
                          n_joints=n_joints)
        self.V = LyapunovNet(n_joints=n_joints, hidden_dim=lyap_hidden)
        self.alpha = alpha

    def forward(self, x):
        return self.f(x)

    def lyapunov(self, x):
        return self.V(x)

    def safe_velocity(self, x, scale_factor=None):
        """Return f(x) projected onto {v : (∇V ⊙ s) · v ≤ -alpha · V},
        where s = scale_factor accounts for the difference between the model
        output v_n and the actual rate dx_n/dt.

        Background: V is defined on the normalised state x_n = e/state_std,
        but the model outputs v_n = q̇/vel_scale. The actual time derivative
        of x_n is dx_n/dt = q̇/state_std = v_n ⊙ (vel_scale/state_std). So
        dV/dt = ∇V · dx_n/dt = (∇V ⊙ vel_scale/state_std) · v_n.
        Pass scale_factor = vel_scale/state_std (componentwise) so the
        projection enforces dV/dt ≤ -αV in real time, not just in normalised
        coordinates.
        """
        with torch.enable_grad():
            x_g = x.detach().clone().requires_grad_(True)
            V_val = self.V(x_g)
            grad = torch.autograd.grad(V_val.sum(), x_g)[0]

        if scale_factor is not None:
            gV_eff = grad * scale_factor
        else:
            gV_eff = grad

        with torch.no_grad():
            v_raw = self.f(x)
            dot   = (gV_eff * v_raw).sum(dim=-1, keepdim=True)
            bound = -self.alpha * V_val.unsqueeze(-1)

            norm_sq = (gV_eff ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            excess  = (dot - bound).clamp(min=0.0)
            v_safe  = v_raw - (excess / norm_sq) * gV_eff
        return v_safe


# ── Loss functions ───────────────────────────────────────────────────────────
def imitation_loss(model, x, q_dot_demo):
    return F.mse_loss(model(x), q_dot_demo)


def stability_loss(model, x, alpha=1.0, scale_factor=None):
    """Enforce dV/dt + alpha · V <= 0 on the training distribution.

    scale_factor = vel_scale/state_std rescales gV so the constraint is on
    the actual dV/dt, not on the dot product in normalised coordinates.
    """
    x_g = x.detach().clone().requires_grad_(True)
    V_val = model.V(x_g)
    grad  = torch.autograd.grad(V_val.sum(), x_g, create_graph=True)[0]

    if scale_factor is not None:
        gV_eff = grad * scale_factor
    else:
        gV_eff = grad

    v     = model.f(x_g)
    dV_dt = (gV_eff * v).sum(dim=-1)

    return F.relu(dV_dt + alpha * V_val).mean()


def total_loss(model, x, q_dot, alpha=1.0, lambda_stab=0.5, scale_factor=None):
    L_imit = imitation_loss(model, x, q_dot)
    L_stab = stability_loss(model, x, alpha, scale_factor=scale_factor)
    return L_imit + lambda_stab * L_stab, L_imit.item(), L_stab.item()
