# Neural Dynamical System

How the joint-space Neural DS in `src/neural_ds.py` is built, what it computes,
and why each piece is there.

## State and goal

The DS operates in joint space, on the **error coordinate**

```
e = q - q_goal       q, q_goal ∈ R^7
```

`q` is the current Franka arm configuration (7 joints; fingers are handled
separately by the gripper). `q_goal` is the joint configuration produced by a
single Lula IK call at each primitive transition. While the primitive is
running, `reach` and `transport` are driven by the DS. The short constrained
primitives, `grasp`, `lift`, and `place`, use the same `q_goal` convention but
are executed by the Lula joint-space controller instead of learned DS
checkpoints.

Using `e` (rather than concatenating `[q, q_goal] ∈ R^14`) is deliberate: the
distribution of `e` is the same regardless of which IK solver computed
`q_goal`, so we don't get a train/deploy mismatch when q_goal sits in a
different null-space configuration than what training saw.

## Architecture

Two small MLPs, both over `e ∈ R^7`.

### `NeuralDS` — the velocity field $f_\theta$

```
e (7) ──► Linear(7, 128) ──► tanh ──► Linear(128, 128) ──► tanh ──► Linear(128, 7) ──► q̇ (7)
```

Output is interpreted as a joint velocity command (in normalised units; deploy
code rescales by `vel_scale`).

### `LyapunovNet` — the scalar Lyapunov value $V_\phi$

```
e (7) ──► Linear(7, 64) ──► tanh ──► Linear(64, 64) ──► tanh ──► Linear(64, 64) ──► g(e) (64)
```

The 64-dim output `g(e)` is a **feature map**, not a scalar. The scalar `V` is
built from it (see below).

## The nominal DS

`NeuralDS.forward` (`src/neural_ds.py:44`) computes

$$
\dot q \;=\; f_\theta(e) \;=\; \bigl(\text{net}(e) - \text{net}(0)\bigr) \;-\; k_{\text{skip}}\, e
$$

Two pieces:

1. **Learned residual** $f_\text{res}(e) = \text{net}(e) - \text{net}(0)$.
   By subtracting `net(0)` we force $f_\text{res}(0) = 0$ exactly.
2. **Linear prior** $-k_{\text{skip}} e$ (controlled by
   `stable_skip_gain`). When non-zero, this adds a globally-attracting linear
   field around which the residual learns. The current default is `1.0`, so
   the learned residual is trained around a stable linear joint-error field.

Either way, the equilibrium at `e = 0` is **guaranteed by construction**:
$f_\theta(0) = 0$ regardless of weights.

### Why subtract `net(0)`

#### What `net(0)` actually is

`net(0)` is just `self.net` (the `nn.Sequential` defined in
`src/neural_ds.py:36-42`) called on the all-zeros input vector
`[0, 0, 0, 0, 0, 0, 0]`. It returns a 7-dim vector, exactly the same shape
as any other forward pass. The literal code is three lines
(`src/neural_ds.py:48-50`):

```python
zero = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
residual = self.net(x) - self.net(zero)
return residual - self.stable_skip_gain * x
```

#### What it does, concretely

Imagine we just trained the network and we ask: "if the arm is exactly at the
goal (`e = q − q_goal = 0`), what velocity should the DS command?"

The answer should obviously be **zero** — the arm is at the goal, it should
stop.

But a plain MLP doesn't give zero. Each `Linear` layer is `Wx + b`. Feeding it
`x = 0`:

- Layer 1: `W₁·0 + b₁ = b₁`     (just the bias)
- `tanh(b₁)`
- Layer 2: `W₂·tanh(b₁) + b₂`   (some other vector)
- `tanh(...)`
- Layer 3: `W₃·tanh(...) + b₃`

The output is some 7-dim vector that depends on every weight and bias in the
network. There's no reason it would be zero. After training it might be
*small* (the imitation loss saw demonstrations end near the goal with
velocity ≈ 0), but it won't be exactly zero, and "small" isn't good enough:
constant non-zero velocity at the goal means the arm drifts past it forever.

#### The fix

Define the DS as `net(e) − net(0)` instead. Plug in `e = 0`:

```
f(0) = net(0) − net(0) = 0       ← exactly zero, every time
```

Doesn't matter what the weights are, what the biases are, whether the network
is trained or randomly initialised, whether the demonstrations were
perfect — the output at `e = 0` is mathematically guaranteed to be the zero
vector, because we're subtracting a number from itself.

#### A tiny analogy

This is the same pattern as: "I want a function `h(x)` such that `h(5) = 0`."
For any function `f`, define

```
h(x) = f(x) − f(5)
```

and `h(5) = 0` is automatic. It's not learned, it's not approximate, it's
just arithmetic. `net(e) − net(0)` is the same trick applied to a network:
take whatever the MLP outputs, subtract its value at the point you want to be
the zero, and you've shifted the function so that point is now exactly zero.

#### Why this matters here

Subtracting `net(0)` makes `f_θ(0) = 0` a **structural property of the
architecture**, not a training outcome. The imitation loss only has to learn
the *shape* of the velocity field; the equilibrium at the goal is given for
free. The same trick is used by `LyapunovNet` (subtract `g(0)`) to guarantee
`V(0) = 0`.

It costs one extra forward pass through the MLP per call, on the 7-dim zero
vector — negligible.

## The Lyapunov function

`LyapunovNet.forward` (`src/neural_ds.py:74`) computes

$$
V(e) \;=\; \lVert g_\phi(e) - g_\phi(0) \rVert^2 \;+\; \varepsilon\,\lVert e \rVert^2
$$

Two terms, both zero at `e = 0` and strictly positive elsewhere:

- $\lVert g(e) - g(0) \rVert^2$ — squared norm of the learned feature map's
  deviation from its value at the goal. PSD by construction; the network
  shapes the function over the workspace.
- $\varepsilon \lVert e \rVert^2$ with $\varepsilon = 0.5$ — a quadratic
  regulariser that keeps `V` from going flat far from the origin and
  guarantees positive definiteness even if `g` learns something pathological.

`V` is therefore a valid Lyapunov candidate **by construction**, with no
learned PSD parameterisation needed.

## Training

The current pipeline trains DS checkpoints only for `reach` and `transport`.
Demonstrations still contain labels for `grasp`, `lift`, and `place` so those
segments can be audited, but the deployment controller executes them with Lula.
Collection is Lula-only, slower than the early debug runs, includes non-recorded
settling pauses between primitives, and uses the same dynamic transport stack
clearance that deployment uses.

`total_loss` (`src/neural_ds.py:162`) combines two terms:

$$
\mathcal L \;=\; \underbrace{\bigl\lVert f_\theta(e) - \dot q_\text{demo} \bigr\rVert^2}_{\text{imitation}}
\;+\; \lambda_\text{stab}\,
\underbrace{\bigl[\,\dot V(e) + \alpha\, V(e)\,\bigr]_+}_{\text{stability hinge}}
$$

- **Imitation loss.** MSE against demonstrated joint velocities.
- **Stability hinge.** Enforces $\dot V + \alpha V \leq 0$ softly on the data
  distribution. `dV/dt` is computed via autograd as $\nabla V \cdot \dot q$.
  `α` controls the required exponential decay rate of `V`.
- **Scale factor.** `stability_loss` (`src/neural_ds.py:141`) takes a
  `scale_factor = vel_scale / state_std` so the constraint is enforced on the
  real-time `dV/dt`, not the dot product in the network's normalised
  coordinates.

`λ_stab` (default `0.05` in `configs/default.yaml`) trades off imitation
fidelity against stability margin.

## Safe velocity (hard projection at deploy)

`StableNeuralDS.safe_velocity` (`src/neural_ds.py:102`) is opt-in via
`--use_safe` at deployment. It takes the nominal `v = f_θ(e)` and projects it
onto the half-space where the Lyapunov constraint holds **exactly**:

$$
v_\text{safe} \;=\; v \;-\; \frac{\bigl[\,\nabla V \cdot v + \alpha V\,\bigr]_+}{\lVert \nabla V \rVert^2}\,\nabla V
$$

- If the soft training constraint already holds at `e`, the `[·]_+` is zero
  and `v_safe = v` — no modification.
- Otherwise, just enough of the offending component along `∇V` is subtracted
  to satisfy `∇V · v_safe ≤ -αV`.

The same `scale_factor = vel_scale / state_std` is applied to `∇V` here so
the projection enforces real-time `dV/dt`, not the normalised version.

This gives us two ablation modes for the writeup:

- Soft stability only — `--use_safe` off. The training loss encouraged
  `dV/dt + αV ≤ 0`, but nothing enforces it at runtime.
- Hard stability — `--use_safe` on. Lyapunov decrease is guaranteed at every
  step, at the cost of deviating from the imitation field when the two
  conflict.

## Summary of guarantees

| Property                           | How it's guaranteed                                |
|------------------------------------|----------------------------------------------------|
| $f_\theta(0) = 0$ (equilibrium)    | `net(e) - net(0)` subtraction, structural          |
| $V(e) > 0$ for $e \neq 0$, $V(0)=0$ | `g(e) - g(0)` plus $\varepsilon\lVert e\rVert^2$, structural |
| $\dot V + \alpha V \leq 0$         | Soft (training loss) or hard (`safe_velocity`)     |

The first two are architectural; the third is the actual learning problem.
