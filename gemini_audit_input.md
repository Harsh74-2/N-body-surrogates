# Thesis audit digest (2026-09-23)

Purpose: independent cross-check of every hand-written numeric claim in the thesis against the generated macro values (metrics.tex, 1520 macros generated from the final retrain artifacts). Claims quoted verbatim from thesis.tex / thesis_appendix_results.tex; the value lines are the macro contents.

## Claim 1 — OOD survivor means (tab:ood) / tab:stable-vs-single
Thesis: 'the stable variant lands within five points of its sibling in 10 of the 11 comparable cells'; 'stability-trained GNN degrades from X to Y at N=100, while the stability-trained LSTM is the only variant with any surviving OOD preset there at all'.
Values (survivor mean %, single -> stable, delta in pp):
  Mlp  10: single=66.6  stable=65.0  delta=-1.6
  Mlp  25: single=68.9  stable=68.7  delta=-0.2
  Mlp  50: single=64.6  stable=64.6  delta=+0.0
  Mlp 100: single=65.6  stable=66.6  delta=+1.0
  Lstm  10: single=64.7  stable=63.6  delta=-1.1
  Lstm  25: single=70.8  stable=74.3  delta=+3.5
  Lstm  50: single=64.0  stable=65.2  delta=+1.2
  Lstm 100: single=None  stable=97.9  delta=n/a
  Gnn  10: single=69.3  stable=69.1  delta=-0.2
  Gnn  25: single=66.7  stable=66.6  delta=-0.1
  Gnn  50: single=73.0  stable=73.1  delta=+0.1
  Gnn 100: single=72.2  stable=82.5  delta=+10.3

## Claim 2 — 'stable beats sibling on single-step MSE in 8 of 12 pairs'
  Mlp  10: single=7.79e-08  stable=8.28e-08  stable-wins=False
  Mlp  25: single=7.75e-08  stable=8.29e-08  stable-wins=False
  Mlp  50: single=1.2e-07  stable=1.3e-07  stable-wins=False
  Mlp 100: single=2.15e-08  stable=1.91e-08  stable-wins=True
  Lstm  10: single=1.47e-07  stable=1.51e-07  stable-wins=False
  Lstm  25: single=1.2e-08  stable=1.16e-08  stable-wins=True
  Lstm  50: single=2.26e-06  stable=1.84e-06  stable-wins=True
  Lstm 100: single=5.3e-08  stable=4.77e-08  stable-wins=True
  Gnn  10: single=2.71e-06  stable=2.55e-06  stable-wins=True
  Gnn  25: single=1.51e-06  stable=1.15e-06  stable-wins=True
  Gnn  50: single=8.38e-08  stable=8.05e-08  stable-wins=True
  Gnn 100: single=2.83e-07  stable=4.04e-08  stable-wins=True

## Claim 3 — 'improves the K=128 rollout slope in 10 of 12 (two exceptions: GNN at N=10 and a 0.06% tie at the LSTM's N=50)'
  Mlp  10: single=1.1454e-03  stable=9.2212e-04  stable-better=True
  Lstm  10: single=1.0477e-03  stable=7.5580e-04  stable-better=True
  Gnn  10: single=8.4378e-06  stable=1.0657e-05  stable-better=False
  Mlp  25: single=1.7637e-04  stable=1.6162e-04  stable-better=True
  Lstm  25: single=3.1018e-04  stable=2.9448e-04  stable-better=True
  Gnn  25: single=1.8886e-05  stable=1.3701e-05  stable-better=True
  Mlp  50: single=1.1790e-04  stable=9.8293e-05  stable-better=True
  Lstm  50: single=1.3646e-04  stable=1.3653e-04  stable-better=False
  Gnn  50: single=1.7194e-06  stable=1.5360e-06  stable-better=True
  Mlp 100: single=1.9257e-04  stable=1.1763e-04  stable-better=True
  Lstm 100: single=4.8084e-02  stable=2.6177e-02  stable-better=True
  Gnn 100: single=1.6719e-05  stable=1.0246e-06  stable-better=True
  LSTM N=50 relative gap: 0.057% (thesis says 0.06% tie)

## Claim 4 — 'GNN lowest rollout slope at every N, one to two orders of magnitude below MLP/LSTM'; 'stable GNN better from N=25 on'; 'GNN_st N=100 slope ~16x below sibling'
  N= 10: best=gnn (8.438e-06)  GNN=8.438e-06  GNN_st=1.066e-05  MLP=1.145e-03  LSTM=1.048e-03
  N= 25: best=gnn_st (1.370e-05)  GNN=1.889e-05  GNN_st=1.370e-05  MLP=1.764e-04  LSTM=3.102e-04
  N= 50: best=gnn_st (1.536e-06)  GNN=1.719e-06  GNN_st=1.536e-06  MLP=1.179e-04  LSTM=1.365e-04
  N=100: best=gnn_st (1.025e-06)  GNN=1.672e-05  GNN_st=1.025e-06  MLP=1.926e-04  LSTM=4.808e-02
  GNN N=100 sibling/stable slope ratio: 16.3x

## Claim 5 — GNN 1e-8 single-step band wording
Thesis (interpretation): 'at N_train=50 it reaches the 1e-8 single-step band of the MLP (resGnnMseNfifty, the best GNN cell of the sweep), with its stability-trained sibling holding that band at N=100 (resGnnStMseNhundred)'.
(Note the claim is deliberately NOT made for the single-step GNN at N=100.)
  Mlp  10: 7.79e-08
  Mlp  25: 7.75e-08
  Mlp  50: 1.2e-07
  Mlp 100: 2.15e-08
  MlpSt  10: 8.28e-08
  MlpSt  25: 8.29e-08
  MlpSt  50: 1.3e-07
  MlpSt 100: 1.91e-08
  Lstm  10: 1.47e-07
  Lstm  25: 1.2e-08
  Lstm  50: 2.26e-06
  Lstm 100: 5.3e-08
  LstmSt  10: 1.51e-07
  LstmSt  25: 1.16e-08
  LstmSt  50: 1.84e-06
  LstmSt 100: 4.77e-08
  Gnn  10: 2.71e-06
  Gnn  25: 1.51e-06
  Gnn  50: 8.38e-08
  Gnn 100: 2.83e-07
  GnnSt  10: 2.55e-06
  GnnSt  25: 1.15e-06
  GnnSt  50: 8.05e-08
  GnnSt 100: 4.04e-08

## Claim 6 — single-step error bands (sec:res:ood)
Thesis: 'single-step mean error on the disc is 0.00-0.03% of L for all three architectures at every count, against 0.3-29.9% on the OOD presets (inner_planets the high end, solar_system_extended the low end)'
Computed: disc range 0.00-0.00;  OOD range 0.30-29.90
  disc min cell:  0.0
  OOD max = inner_planets? (29.9, 'Ip', 'Gnn', 'Nfifty')
  OOD min = solar_system_extended? (0.3, 'Sse', 'Gnn', 'Nten')

## Claim 7 — jupiter_galileans no-compounding band 'all three architectures sit in a ~10.7-11.4% band across N'
Computed jg range: 10.70-11.40  values: [10.7, 10.8, 11.1, 11.2, 11.3, 11.4]

## Claim 8 — latency appendix
Thesis: 'solver O(N^2) from 0.029 ms at N=10 to 1.62 ms at N=200; MLP crossover only at N*=200 single-frame / N*=100 batched; LSTM above the solver at every count; GNN slowest surrogate at every N'
  solver N=10: 0.02939 ms
  solver N=25: 0.07604 ms
  solver N=50: 0.1736 ms
  solver N=100: 0.487 ms
  solver N=200: 1.616 ms
  mlp surrogate_single_frame: N10=0.4726  N25=0.4782  N50=0.663  N100=0.8348  N200=1.112
  lstm surrogate_single_frame: N10=1.616  N25=2.287  N50=4.799  N100=6.26  N200=8.468
  gnn surrogate_single_frame: N10=4.686  N25=10.19  N50=27.93  N100=114.9  N200=436.8
  mlp surrogate_batched_amortised: N10=0.03677  N25=0.09893  N50=0.2017  N100=0.4652  N200=0.9477
  lstm surrogate_batched_amortised: N10=0.5008  N25=1.258  N50=2.988  N100=5.582  N200=10.44
  gnn surrogate_batched_amortised: N10=1.218  N25=7.366  N50=31.24  N100=117.3  N200=476.9

