## Task(3) BF kernel regression

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