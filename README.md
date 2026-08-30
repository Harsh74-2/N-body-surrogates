# N-body-surrogates

Per-body, variable-N machine-learning surrogates for N-body gravitational
dynamics. Three architectures (MLP, LSTM, GNN) are trained on synthetic galaxy
discs and then tested out of distribution on real Solar-System initial
conditions, with a rollout-stability benchmark that contrasts single-step
training against a stability-trained variant.

The goal is to measure how well a learned force model trained on one
distribution (rotating discs of roughly equal-mass particles) transfers to a
very different one (the Solar System, with mass ratios of ~10⁵ and Keplerian
orbits), and to compare which architecture generalises furthest.

---

## Why

Direct N-body integration is expensive. A leapfrog or symplectic integrator
has to resolve the shortest dynamical timescale in the system and take many
small steps, so long horizons and large N become costly. A learned surrogate
that predicts the next state in one forward pass can act as a fast evaluator,
trading exactness for speed.

The catch is generalisation. A surrogate trained on one family of systems will
not necessarily transfer to another, and it is not obvious which architecture
handles that transfer best. This project sets up a controlled comparison:

- three architectures with the same per-body, shared-weight design,
- a fixed training distribution and a fixed training objective,
- an out-of-distribution test bed built from real Solar-System data,
- a stability benchmark that separates short-horizon accuracy from
  long-horizon divergence.

The per-body formulation is what makes the comparison clean. Because each body
is processed by the same shared weights, the parameter count does not depend on
*N*. A model trained at one body count can be evaluated at another without
retraining, which is exactly the transfer we want to measure.

---

## What it is

### The three surrogates

| Model | Hidden / layers | Trainable params | Role |
|---|---|---|---|
| MLP  | 256 / depth 4      | 210,182 | Feed-forward baseline, no spatial structure. |
| LSTM | 256 / 2 layers     | 865,542 | Sequence model over the input window. |
| GNN  | 128 / 2 msg passes | 167,815 | Message-passing graph net, body count agnostic. |

All three take a per-body state (position, velocity, and mass-derived features)
and predict the next state. The MLP and LSTM consume a sliding window of
*W* = 5 frames; the GNN operates on a single frame with one edge per body pair
and so sees the pairwise structure the MLP and LSTM miss.

Parameter counts are independent of *N* by construction. Only the
per-forward-pass compute scales with body count, and for the GNN that scaling is
*N*² because of the all-pairs messages.

### Training distribution

Models are trained on synthetic galaxy discs produced by `init_galaxy_disc`:
*N* = 25 bodies drawn from a real initial-mass function (masses log-uniform in
roughly 0.1 to 50 solar masses), placed on a rotating disc with a Plummer
softening length. The disc is integrated in dimensionless N-body units where
*G* = 1, the total mass sums to 1, the characteristic length is 1, and the time
unit is *T* = √(*L*³ / (*G*·*M*)). The single-step training objective is a
mean-squared error on the predicted state, optionally combined with a
rollout-energy-drift term (see the stability variant below).

### The stability-trained variant

Each model is also retrained with the rollout-energy term enabled
(`w_rollout = 0.1`). This backprops through a short rollout and penalises
energy drift, which should improve long-horizon stability. The contrast between
the single-step checkpoint and the stability-trained checkpoint is the core of
the stability benchmark.

---

## How it works

### Single-step training and evaluation

`scaling_sweep.py` trains all three models across *N* ∈ {10, 25, 50, 100} on
the disc datasets, all with the same single-step objective and the same
architectures. `evaluate_models.py` then scores each checkpoint with
single-step MSE and a rollout energy-drift metric.

### Rollout stability benchmark

`stability_benchmark.py` runs a *K* = 128 step autoregressive rollout per
checkpoint against the true trajectory and records per-step metrics:

- position, velocity, and full-state MSE versus ground truth,
- energy drift relative to the initial predicted energy,
- a composed loss (MSE plus a weighted energy term),
- the slope of each metric versus step (the stability gradient),
- the first divergence step, if the rollout blows up.

The MLP and LSTM use a sliding-window rollout seeded with the true *W* = 5
window, matching how they were trained. The GNN uses its native one-step
interface. This is a deliberate methodological choice: to measure position and
velocity error against ground truth, the rollout has to feed the model an
in-distribution input, which means shifting the window forward each step.

Six checkpoints are benchmarked per *N* (three single-step plus three stable)
at *N* ∈ {10, 25, 50}; *N* = 100 is excluded from the stability benchmark only,
because backprop through *K* steps at *N* = 100 is memory-bound.

### Real-case out-of-distribution validation

`real_case_validation/` takes the disc-trained checkpoints and runs them on
real Solar-System initial conditions. A real system is rescaled into the same
dimensionless N-body units the surrogates expect: the mass unit is the total
system mass, the length unit is a characteristic radius (the outermost orbit,
or an explicit override for satellite systems), and the time unit follows from
*T* = √(*L*³ / (*G*·*M*)). The centre-of-mass velocity is stripped so the
network sees a zero-momentum state, as it was trained on.

A high-precision leapfrog integrator provides the ground-truth reference
trajectory. A Kepler's-third-law check on that reference confirms it is a
faithful integrator on these initial conditions: deviation is below 0.1% for
the inner bodies. Bodies whose period is longer than the simulation window
show NaN, as expected.

---

## Repository layout

```
pipeline_config.py          Constants, physical units, model and feature config
losses.py                   Single-step loss, total energy, rollout energy loss
scaling_sweep.py            Trains all 3 models across N in {10, 25, 50, 100}
mlp_train.py                MLP trainer
lstm_train.py               LSTM trainer
gnn_train.py                GNN trainer
evaluate_models.py          Single-step and rollout evaluation, parameter count
stability_benchmark.py      K-step rollout stability: per-step metrics and slopes
train_stable_variants.sh    Retrains all 3 models with w_rollout=0.1 at N in {10,25,50}
real_case_validation/       Solar-System OOD validation package and reports
plots/                      Reproducibility figures (eval, scaling, stability)
results/                    JSON metrics and stability reports
```

Checkpoints (`training_runs/*.pt`), raw datasets (`raw_data/`,
`ml_ready_data/`), the bulky per-preset plot directories, and all thesis
documents are not committed. The committed reports and figures form the
verification bundle and can be regenerated from the scripts.

---

## Setup

Python 3.11 or newer. CUDA is optional for evaluation and real-case validation
but strongly recommended for training.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch numpy scipy matplotlib
```

There is no packaged install. Run the scripts from the repository root so the
`pipeline_config` and `real_case_validation` imports resolve.

---

## Instructions

### 1. Train the three models across body counts

```bash
python scaling_sweep.py
```

This produces single-step checkpoints under
`training_runs/N{N}/{mlp,lstm,gnn}/` for *N* ∈ {10, 25, 50, 100}.

### 2. Evaluate

```bash
python evaluate_models.py
```

Reports single-step MSE, rollout energy drift, and the parameter count per
model.

### 3. Train the stability variants and run the stability benchmark

```bash
bash train_stable_variants.sh

for N in 10 25 50; do
  python stability_benchmark.py \
    --ckpt training_runs/N${N}/mlp/model_best.pt:mlp \
    --ckpt training_runs/N${N}/lstm/model_best.pt:lstm \
    --ckpt training_runs/N${N}/gnn/model_best.pt:gnn \
    --ckpt training_runs/N${N}/mlp_stable/model_best.pt:mlp \
    --ckpt training_runs/N${N}/lstm_stable/model_best.pt:lstm \
    --ckpt training_runs/N${N}/gnn_stable/model_best.pt:gnn \
    --N ${N} --K 128 --json results/N${N}/stability.json --out plots
done

python stability_benchmark.py --aggregate-only --out plots
```

This writes per-N stability JSON and plots, plus an overview plot of the slopes
versus N.

### 4. Run the real-case OOD validation

```bash
python -m real_case_validation.real_case_runner \
  --ckpt training_runs/N25/gnn/model_best.pt:gnn:GNN_N25 \
  --ckpt training_runs/N25/mlp/model_best.pt:mlp:MLP_N25 \
  --ckpt training_runs/N25/lstm/model_best.pt:lstm:LSTM_N25 \
  --out real_case_validation/report
```

The `--ckpt` flag takes `path:type` or `path:type:label`. The optional label
disambiguates same-type checkpoints in the combined report (for example
`MLP_N25` versus `MLP_N25_stable`). Omit `--preset` to run all built-in
presets, or pass `--preset <name>` to run a subset.

The runner writes a markdown report, a JSON report, and a dashboard PNG, plus a
per-preset directory with trajectory and energy plots.

---

## Real-case presets

| Preset | N | What it probes |
|---|---|---|
| `inner_planets` | 5 | Mercury through Mars at an in-distribution timestep. |
| `full_solar_system` | 9 | All 8 planets. The headline OOD case. |
| `sun_earth_only` | 2 | Two-body Keplerian reference. |
| `sun_planets_moon` | 10 | Adds Earth's Moon, a tight satellite orbit. |
| `jupiter_galileans` | 5 | Planetocentric Jupiter plus 4 Galilean moons. |
| `solar_system_extended` | 19 | Planets, Moon, 5 dwarfs, 4 Galilean moons: the limit. |
| `disc_imf_in_distribution_baseline` | 25 | In-distribution sanity check. |

The satellite presets use a planetocentric frame where appropriate and take the
largest satellite orbit as the characteristic length, so every body sits at
O(1) in N-body units and the surrogate receives a properly scaled input.

---

## Tests

Three layers of testing, each in its own script:

- **Single-step and rollout evaluation** (`evaluate_models.py`): per-model MSE
  and energy drift on the training distribution, plus a parameter-count check.
- **Rollout stability** (`stability_benchmark.py`): per-step MSE, energy drift,
  and slopes over a *K* = 128 horizon, with divergence detection. The
  in-distribution disc baseline confirms the pipeline behaves correctly before
  any OOD claim is made.
- **Real-case OOD validation** (`real_case_validation/`): position and state
  MSE, max and mean error normalised by the characteristic length, frames to
  half-length error, and max energy drift, for every checkpoint across every
  preset. A Kepler's-third-law check on the reference integrator validates the
  ground truth itself.

---

## Results and conclusions

The numbers below are representative; exact per-checkpoint, per-preset values
are in `real_case_validation/report_*/real_case_report.md` and
`results/*/stability.json`.

- **The GNN generalises best out of distribution.** On `full_solar_system` the
  GNN trained at *N* = 10 reaches a position MSE of about 0.07, where the MLP
  and LSTM stay in the single-digit range. The GNN's pairwise message passing
  is the only architecture that sees the spatial structure the others miss,
  and it shows in the transfer.
- **The GNN's error is nearly flat across the training body count.** A
  smaller-*N* checkpoint works about as well as a larger-*N* one on the OOD
  presets, so for the GNN the training body count is not the thing that limits
  transfer.
- **Stability training helps in distribution but hurts out of distribution.**
  The rollout-energy term flattens the in-distribution stability curves, but on
  the OOD presets it degrades the GNN substantially (for example
  `full_solar_system` goes from about 0.07 to roughly 2.2). Optimising for
  energy conservation on the training distribution trades away generalisation.
- **Body count is not the limiting factor at the limit preset.** On
  `solar_system_extended` (19 bodies, aligned to the same horizon as
  `sun_planets_moon`), the GNN still places bodies well at the smaller training
  counts. The limiting factor is the mixed scale and the mass hierarchy, not N.
- **The in-distribution baseline passes.** The 25-body disc sanity check
  confirms the OOD failures come from distribution shift, not a pipeline bug.
- **The reference is faithful.** Kepler's third law holds to below 0.1% for the
  inner bodies, so the ground truth the surrogates are scored against is
  trustworthy.

The overall conclusion: for per-body N-body surrogates that must transfer
across body count and across system family, a message-passing GNN trained with
a plain single-step objective is the best of the three options tested, and
adding rollout-aware stability training is a net negative for transfer even
though it helps in distribution.

---

## Applicability and limitations

### Where this is useful

- **Fast approximate trajectory propagation** where a full integrator is too
  expensive and some error is acceptable, for example preview renders,
  interactive visualisations, or initial scoping before a precise integration.
- **Systems with uncertain body count.** The per-body, shared-weight design
  means a single trained model applies across *N*, so the deployment body
  count does not need to be known at training time.
- **Surrogate-based screening.** Run the cheap surrogate to flag
  configurations worth integrating precisely, then spend the integrator budget
  only there.

### Limitations

- The surrogates are trained on non-Keplerian discs, so they do not learn
  Kepler's third law and will not conserve orbital structure the way a
  symplectic integrator does. They are approximators, not replacements for a
  faithful integrator in any setting that demands long-term energy
  conservation.
- Transfer degrades on extreme mass hierarchies (mass ratios of ~10⁵ and up)
  and on mixed-scale systems such as `solar_system_extended`.
- The sliding-window training split (*W* = 5, stride = 1) leaks information
  between train and validation windows. This is a known property of the dataset
  construction and is not corrected here.
- Stability training uses an energy-drift loss only, with no ground-truth MSE
  term in the rollout objective, so it improves conservation at the cost of
  positional accuracy on the training distribution as well.

---

## Methodology notes

- The stability benchmark's sliding-window rollout for the MLP and LSTM differs
  from the degenerate-window rollout in `evaluate_models.py` (which feeds the
  same frame five times to isolate energy drift). The stability benchmark
  measures position and velocity error against ground truth, so it must feed an
  in-distribution window, which means shifting the window forward each step.
- Stability training backprops through *K* steps with no detach, which is
  O(*K*) in memory and several times slower than the single-step run. Training
  batch sizes are halved accordingly, and *K* is kept small (5) during training.
- Real-case rescaling reports its chosen mass and length units in each
  per-preset summary so the mapping back to SI is auditable.

---