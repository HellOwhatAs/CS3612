## Task(1) Gradient descent on linear model
$$
L(\beta) = {\lVert Y - X\beta\rVert}_2^2
$$
$$\begin{aligned}
    \frac{\partial L(\beta)}{\partial \beta} &= \frac{\partial {\lVert Y - X\beta\rVert}_2^2}{\partial \beta}\\
    &= -2 X^\top\left(Y-X\beta\right)
\end{aligned}$$

## Task(2) Ridge regression

## Task(3) RBF kernel regression

$$K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$$

$$\begin{aligned}
    K &= \begin{bmatrix}
            \phi(x_1)^\top\phi(x_1) & \cdots & \phi(x_1)^\top\phi(x_n)\\
            \vdots & \ddots & \vdots\\
            \phi(x_m)^\top\phi(x_1) & \cdots & \phi(x_m)^\top\phi(x_n)
        \end{bmatrix} \\
    &=  \begin{bmatrix}
            \phi(x_1)^\top\\
            \vdots\\
            \phi(x_n)^\top
        \end{bmatrix}
        \begin{bmatrix}
            \phi(x_1) & \cdots & \phi(x_n)
        \end{bmatrix}
\end{aligned}$$

$$\begin{aligned}
    Y_\text{train}^\text{pred} &= Kc\\
    &= \begin{bmatrix}
    \phi(x_1)^\top\\
    \vdots\\
    \phi(x_n)^\top
\end{bmatrix} \begin{bmatrix}
    \phi(x_1) &
    \cdots &
    \phi(x_n)
\end{bmatrix} \begin{bmatrix}
    c_1\\
    \vdots\\
    c_n
\end{bmatrix}\\
    &= \begin{bmatrix}
    \phi(x_1)^\top\\
    \vdots\\
    \phi(x_n)^\top
\end{bmatrix}\sum_{i=1}^n c_i \phi(x_i)
\end{aligned}$$

$$\begin{aligned}
    Y_\text{test}^\text{pred} &= \begin{bmatrix}
    \phi(x_1)^\top\\
    \vdots\\
    \phi(x_m)^\top
\end{bmatrix}\sum_{i=1}^n c_i \phi(x_i)\\
    &= \begin{bmatrix}
    \phi(x_1)^\top\\
    \vdots\\
    \phi(x_m)^\top
\end{bmatrix} \begin{bmatrix}
    \phi(x_1) &
    \cdots &
    \phi(x_n)
\end{bmatrix} \begin{bmatrix}
    c_1\\
    \vdots\\
    c_n
\end{bmatrix}\\
    &=\begin{bmatrix}
        \phi(x_1)^\top\phi(x_1) & \cdots & \phi(x_1)^\top\phi(x_n)\\
        \vdots & \ddots & \vdots\\
        \phi(x_m)^\top\phi(x_1) & \cdots & \phi(x_m)^\top\phi(x_n)
    \end{bmatrix} \begin{bmatrix}
    c_1\\
    \vdots\\
    c_n
\end{bmatrix}
\end{aligned}$$

## Task(4)

## Task(5) Lasso regression

$$
L(\beta) = \frac{1}{2}{\lVert Y-X \beta\rVert}_2^2 + \lambda {\lVert\beta\rVert}_1
$$

$$\begin{aligned}
    \frac{\partial L(\beta)}{\partial\beta} &= \frac{\partial \frac{1}{2}{\lVert Y-X \beta\rVert}_2^2 + \lambda {\lVert\beta \rVert}_1}{\partial\beta}\\
    &= \lambda\cdot\text{sign}(\beta) - X^\top \left(Y-X\beta\right)
\end{aligned}$$