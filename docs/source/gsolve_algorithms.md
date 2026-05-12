# Methodological framework to transform relative gravity observations to gravity values

Reilly [1] gives the relative gravimeter reading {math}`x_{i, j}`, made at location {math}`j` and time {math}`t_i` as

```{math}
:label: McCubbine1
 x_{i,j} = g_j − a − b(t_i − t_1) + z_{i, j} − e_i
 ```

where,

* {math}`g_j` is the true gravity value at location {math}`j`,

* {math}`−b(t_i − t_1)` is the value corresponding to the amount meter has drifted by time {math}`t_i`,

* {math}`z_{i, j}` is the tidal gravitational effect at location {math}`j` and time {math}`t_i` (this is a known quantity that can be computed by the
Longman [3] formula),

* {math}`a` is the gravimeter zero point (i.e. the absolute gravity
value where the relative gravimeters dial reads zero, in the
absence of any drift or tidal effects), and

* {math}`e_i` is a (small) error term associated to the meter reading at
time {math}`t_i`.

Collecting the known quantities and the error term on the right hand side gives an observation equation of the following form,

```{math}
:label: McCubbine2
g_j − a − b(t_i − t_1) = x_{i,j} − z_{i,j} − e_i.
```

Implicit in all surveys using relative gravimeters, there is a subset of the {math}`N` observations at {math}`M` locations where there are previously
recorded gravity values {math}`\hat g_j` . With respect to the true gravity value
{math}`g_j`, the values {math}`\hat g_j` must also include a (small) error term {math}`\epsilon_j`. This leads to a second equation of the form,

```{math}
:label: McCubbine3
g_j = \hat g_j + \epsilon_j.
```

In order to relate the relative gravity observations to the absolute
gravity values, there must be at least one equation of this type.
i.e. the relative gravity observations must be made at least one location where gravity is already known. The system of linear
equations given by Eqs. (2) and (3) can then be used to determine
the best fitting parameters {math}`\tilde g_j`, {math}`\tilde a` and {math}`\tilde b` to the observations by least squares.

## Method 1: Normal least squares

The system of linear equations
given by Eqs. (2) and (3) can be written in the matrix form given by
Eq. (4).

```{math}
:label: McCubbine4

\begin{vmatrix}
P & \boldsymbol{-1} & \boldsymbol{-T}\\
Q & 0 & 0
\end{vmatrix}
\quad
\begin{vmatrix}
\boldsymbol{g}\\
a\\
b
\end{vmatrix}
\quad
=
\begin{vmatrix}
\boldsymbol{x - z}\\
\boldsymbol{h}
\end{vmatrix}
\quad
+
\begin{vmatrix}
\boldsymbol{e} \\
\boldsymbol{\epsilon}
\end{vmatrix}
```

where,

* **g** is the vector of {math}`g_j`’s
* {math}`P` is an {math}`N × M` matrix with 1’s in the column {math}`j` and row {math}`i` for observations at locations {math}`j` and time {math}`t_i`, with 0’s everywhere else
* **1** is an {math}`N× 1` vector of ones
* {math}`T` is a {math}`N × 1` vector of times {math}`t_i − t1`,
* {math}`Q` is an {math}`M × M` matrix with 1’s on the diagonal for row {math}`j` for locations which correspond to Eq. (3) with 0’s everywhere else
* **x−z** is the vector of meter reading minus the tidal effect for
each observation and
* **h** is a {math}`M × 1` vector with entries {math}`\hat g_i` for which the rows of matrix {math}`Q` have a 1 in them and zero otherwise.

A least squares solution can then be sought which minimizes the
sum of the squared error terms in the vectors **e** and **{math}`\epsilon`**. This will give
the best fitting parameter estimates of the absolute gravity at the surveyed locations {math}`\tilde g`, the gravimeter base line {math}`\tilde a` and the drift factor {math}`\tilde b`.

This method does not decouple the reference sites from the system of observation equations and assumes the errors terms {math}`\epsilon` are non zero. This allows the least squares solution for the absolute
gravity values at the reference sites to differ from the previously recorded values.

```{tip}  Method 1 is suitable in most cases, and should be applied when there is uncertainty about the reference value.
```

## Method 2: Decoupled least squares

Following Reilly [1]. Reilly’s [1]
original method solves Eqs. (2) and (3) by decoupling the reference gravity values from the observation equations.

Decoupling the reference gravity values is performed by modifying Eq. (2). For the observations made at reference gravity sites, the reference gravity values {math}`\hat g_j` are subtracted from the right hand
side of the observation equations and the value {math}`g_j` for that particular location is discarded.  This gives a new observation equation of the
form,

```{math}
:label: McCubbine5
− a − b(t_i − t_1) = x_{i,j} − z_{i,j} − \hat g_j − e_i
```

In terms of Eq. (4), this corresponds to making amendments to
{math}`P` and **x − z**. Entries in column {math}`j` of {math}`P` such that the observation
equations are of the form of (5) are set equal to zero, and the values
{math}`\hat g_j` are subtracted from the vector **x − z**.

This method fixes the reference gravity values so that they do not change in the least squares solution, forcing the residual error terms {math}`\epsilon` to be exactly equal to zero. However the estimated gravity values parameter variances for the reference sites are non-zero, despite the solution fitting exactly to the given previously recorded gravity value.

```{tip}  Method 2 fixes the reference value by removing it from the solution. It is particularly useful when the primary interest is change in gravity over time. In this case the reference value can be set to an arbitrary value, and changes are determined relative to this value.
```

## Method 3: Constrained least squares

This method solves a set of equations of the form of Eq. (1) with constraint equations that
are similar to an error-less form of Eq. (2) using the method of
constrained least squares. The least squares matrix equation is
formulated as follows.

Consider the equation

```{math}
:label: McCubbine6
X\beta = Y − e
```

where, X = [P, −**1**, −**T**],

{math}`\beta` = {math}`\begin{vmatrix} \boldsymbol{g} \\ a \\ b\end{vmatrix}`,

Y = [**x−z**]

with {math}`P`, {math}`T`, {math}`x`, {math}`z`, {math}`\begin{vmatrix} \boldsymbol{g} \\ a \\ b\end{vmatrix}` and {math}`e` are as specified in method 1. Constraints are given by the matrix equation,

```{math}
:label: McCubbine7
K\beta = c
```

where {math}`K` has dimension {math}`k×(M+2)` for some {math}`k`. Here {math}`k` is the number
of reference gravity sites and so {math}`K` has the same form as {math}`Q` but with the zero rows removed. {math}`c` is a {math}`k×1` vector, and contains the specified
gravity values in each row for the location corresponding to the column with a 1 in it in matrix {math}`K`. This equation is similar to an error-less form of Eq. (3). Similar to method 2, the reference gravity
values at locations j are fixed. The aim here is to minimize,

```{math}
:label: McCubbine8
S = (Y − X\beta)^{\prime}(Y − X\beta) + \lambda ^{\prime}(K\beta − c)^2)
```

where {math}`\lambda` is a {math}`k` × 1 vector of Lagrange multipliers.

{math}`S` has a minimum where (1)
{math}`\frac{\delta S}{\delta \beta} = 0`
 and (2) {math}`\frac{\delta S}{\delta \lambda}= 0`.

This is given by,

```{math}
:label: McCubbine9
2X^{\prime}Y − X\lambda) − K^{\prime}\lambda = 0
```

or (for {math}`\hat \lambda = \lambda /2`)

```{math}
:label: McCubbine10
X^{\prime}(Y − X\beta) − K^{\prime} \hat \lambda = 0
```

which implies,

```{math}
:label: McCubbine11
X^{\prime}X\beta − K^{\prime} \hat \lambda = X^{\prime}Y
```

and

```{math}
:label: McCubbine12
K\beta = c
```

This can be written in matrix form as follows,

```{math}
:label: McCubbine13
\begin{vmatrix}
X^{\prime}X & K^{\prime} \\
K & 0
\end{vmatrix}
\quad
\begin{vmatrix}
\beta \\
\hat \lambda
\end{vmatrix}
\quad
=
\quad
\begin{vmatrix}
X^{\prime}Y \\
c
\end{vmatrix}
```

where the best fitting parameters are found by,

```{math}
:label: McCubbine14
\begin{vmatrix}
\beta \\
\hat \lambda
\end{vmatrix}
\quad
=
\begin{vmatrix}
X^{\prime}X & K^{\prime} \\
K & 0
\end{vmatrix}^{-1}
\quad
\begin{vmatrix}
X^{\prime}Y \\
c
\end{vmatrix}
```

The error terms {math}`\epsilon` are considered to be zero and the {math}`\hat g_j` parameter variances are also zero.

```{tip}  Method 3 is most appropriate when the reference gravity values have very low uncertainty, since the solution is adjusted to exactly match the reference values. Any errors in the reference values may distort the solution at the other sites, particularly when there is more than one reference station.
```

## Summary

Three methods to transform relative gravity observations to gravity values have been given. Despite their theoretical differences, in general, each method should return almost identical gravity values. However, the derived gravity value variances may differ between the methods due to the theoretical considerations
of the precision of the reference gravity values.
