# ft linear regression

An introduction to machine learning.

> For this project, you will have to create a program that predicts the price
of a car by using a linear function train with a gradient descent algorithm

## Predictor program

```
python predictor.py
```

This should predict the price of a car based on weights that have been
pretrained. The program will prompt the user for the millage and return 
the estimated price.

```
estimatedPrice(mileage) = t0 + (t1 * mileage)
```

Before training `t0` and `t1` are set to 0, training will find the best fit
values to use.

## Trainer program

```
python trainer.py
```

This will train the model to find the best values of for t0 and t1. The 
downloaded data will be used for the training, it's a 24 entry table of 
km & price.

```
t0 = learningRate * averageError
t1 = learningRate * averageWeightedError
```

_weighted by millage (the single feature in this case)_

This scales the eror by how much impact the millage will have on it.

