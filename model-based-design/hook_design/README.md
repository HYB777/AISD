# Hook Design

The objective in the hook design is to identify an optimal shape that enhances the stiffness
while adhering to specific volume constraints and loads. This can be achieved by minimizing
the strain energy.

## Shape Sampling

We model the half of the hook by lofting 11 2D sections along a guideline, see Appendix G for more details.
For maintaining a naturalistic semblance characterized by smoothness and minimal undulation, 
we apply mean filter to each section once or two at random (each section are generated independently and randomly),
considering its neighboring sections.

Finally, we generate a total of 222,141 shape data points. All data are stored in [data](data).

## High-fidelity Physical Model Learning

We employ the mean-teacher-based active learning algorithm on this dataset, to select 50,000 shapes
for labeling and train the physical model, the corresponding labels are obtained by COMSOL.

Run the following code under the directory [mean-teacher-al](mean-teacher-al).

```
python main.py
```

## Shape Anomaly Detection

Firstly, run the following code to compute the intrinsic dimension.

```
python lpca.py
```

The results should be 68 (round up). We choose 68 as the dimension of the latent space in
the auto-encoder.

Run the following code for training.

```
python main_ae.py --embed-dim 68
```

The trained model are stored in [results_aeID=68](shape-anomaly-detection/results_aeID=68).

## Numerical Optimization

We perform two kinds of optimization problem.

* P1: $\min E$ s.t. $V\leq0.5V_{\text{init}}$
* P2: $\min V$ s.t. $E\leq0.5E_{\text{init}}$

Run the following code

```
python shape_opt_pysparse.py --prob minEconV # problem 1

python shape_opt_pysparse.py --prob minVconE # problem 2
```