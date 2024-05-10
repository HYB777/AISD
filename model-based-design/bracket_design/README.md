# Bracket Design

The objective of bracket design is to identify an optimal cutout shape that
minimizes mass while adhering to an upper stress limit and a lower limit for 
the first natural frequency.

## Shape Sampling

The design of the bracket model is similar to the [Multistudy Optimization of a Bracket](https://comsol.com/model/multistudy-optimization-of-a-bracket-19761),
with the exception that we solely focus on designing the cutout of the bracket. The shape of the cutout
is represented by a uniform periodic cubic B-spline curve.

Similar to [wingrib design](../wingrib_design), for demonstrating the generalisability of our scheme,
we use an overestimated of the control point to represent the cutout, and we use 35 control points. For ensuring these cutouts
exhibit a naturalistic appearance devoid of excessive torsion or self-intersection, we use 8 control points 
to construct the cutouts, which are restricted to disjoint regions, and then fit these 
cutouts with 35 control points, see Appendix H for more details.


At last, we generate a total 244,487 rib data points. The data are stored in [config](config).

## High-fidelity Physical Model Learning

We use the mean-teacher-based active learning algorithm to select 50,000 shapes 
for labeling and training, the selected data would be labelled by COMSOL. For the convenience of the experiment, all data are labelled beforehand, and are stored in [data](data).
Run the following code under [mean-teacher-al](mean-teacher-al)

```
python main.py
```

## Shape Anomaly Detection

Firstly, run the following code to obtain the intrinsic dimension.

```
python lpca.py
```

The result should be 16, that is exactly the dimension of 8 control points. And we choose 24 as the 
dimension of the latent space of the auto-encoder.

Run the following code to train the auto-encoder.
```
python main.py --embed-dim 24
```
The trained model is stored in [shape-anomaly-detection](shape-anomaly-detection).

## Numerical Optimization

Run the following code to perform optimization.

```
python shapeOpt.py
```

The optimized results are stored in [optimized_results](optimized_results).
