"""
Joint-space Neural Dynamical System with learned Lyapunov function.

State:    e = q - q_goal in R^7
Velocity: q_dot = f_theta(e) in R^7

Using joint error as the model state keeps deployment on the same distribution
as training: every primitive starts with a nonzero error and converges to the
origin, regardless of which IK branch produced q_goal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


N_JOINTS = 7
STATE_DIM = N_JOINTS


class NeuralDS(nn.Module):
    """Joint-velocity field f_theta(q - q_goal) -> R^7."""

    def __init__(self, state_dim=STATE_DIM, hidden_dim=128, n_joints=N_JOINTS,
                 stable_skip_gain=0.0):
        super().__init__()
        self.stable_skip_gain = stable_skip_gain
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_joints),
        )

    def forward(self, x):
        # net(x) - net(0) guarantees the learned residual is zero at the goal.
        zero = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
        residual = self.net(x) - self.net(zero)
        return residual - self.stable_skip_gain * x


class LyapunovNet(nn.Module):
    """Positive-definite Lyapunov candidate around e = 0."""

    def __init__(self, n_joints=N_JOINTS, hidden_dim=64, epsilon=0.5):
        super().__init__()
        self.n_joints = n_joints
        self.epsilon = epsilon
        self.g = nn.Sequential(
            nn.Linear(n_joints, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        x_at_goal = torch.zeros_like(x)
        psd = ((self.g(x) - self.g(x_at_goal)) ** 2).sum(dim=-1)
        reg = self.epsilon * (x ** 2).sum(dim=-1)
        return psd + reg


class StableNeuralDS(nn.Module):
    """Joint-space DS with Lyapunov stability machinery."""

    def __init__(self, n_joints=N_JOINTS, hidden_dim=128, lyap_hidden=64,
                 alpha=1.0, stable_skip_gain=0.0):
        super().__init__()
        self.n_joints = n_joints
        self.stable_skip_gain = stable_skip_gain
        self.f = NeuralDS(state_dim=n_joints, hidden_dim=hidden_dim,
                          n_joints=n_joints,
                          stable_skip_gain=stable_skip_gain)
        self.V = LyapunovNet(n_joints=n_joints, hidden_dim=lyap_hidden)
        self.alpha = alpha

    def forward(self, x):
        return self.f(x)

    def lyapunov(self, x):
        return self.V(x)

    def safe_velocity(self, x, scale_factor=None):
        """Project f(x) so dV/dt <= -alpha V in real joint coordinates.

        x is the normalized error e/state_std, while model output is normalized
        by vel_scale. scale_factor=vel_scale/state_std converts the Lyapunov
        gradient to the derivative of normalized error under the real q_dot.
        """
        with torch.enable_grad():
            x_g = x.detach().clone().requires_grad_(True)
            V_val = self.V(x_g)
            grad = torch.autograd.grad(V_val.sum(), x_g)[0]

        gV_eff = grad * scale_factor if scale_factor is not None else grad

        with torch.no_grad():
            v_raw = self.f(x)
            dot = (gV_eff * v_raw).sum(dim=-1, keepdim=True)
            bound = -self.alpha * V_val.unsqueeze(-1)

            norm_sq = (gV_eff ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            excess = (dot - bound).clamp(min=0.0)
            v_safe = v_raw - (excess / norm_sq) * gV_eff
        return v_safe


def imitation_loss(model, x, q_dot_demo):
    return F.mse_loss(model(x), q_dot_demo)


def stability_loss(model, x, alpha=1.0, scale_factor=None):
    """Enforce dV/dt + alpha * V <= 0 on the training distribution."""
    x_g = x.detach().clone().requires_grad_(True)
    V_val = model.V(x_g)
    grad = torch.autograd.grad(V_val.sum(), x_g, create_graph=True)[0]
    gV_eff = grad * scale_factor if scale_factor is not None else grad

    v = model.f(x_g)
    dV_dt = (gV_eff * v).sum(dim=-1)

    return F.relu(dV_dt + alpha * V_val).mean()


def total_loss(model, x, q_dot, alpha=1.0, lambda_stab=0.5, scale_factor=None):
    L_imit = imitation_loss(model, x, q_dot)
    L_stab = stability_loss(model, x, alpha, scale_factor=scale_factor)
    return L_imit + lambda_stab * L_stab, L_imit.item(), L_stab.item()
