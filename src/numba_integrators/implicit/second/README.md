# Implicit solvers, second order

## Solving

$$
y'' = f(x, y, y')
$$

$$
\begin{aligned}
    y_{n+1} & = y_n + \Delta x \cdot {y'}_{n+1} +  {\Delta x}^2 \cdot {y''}_{n+1} / 2 \\
    y'_{n+1} & = y'_n + {\Delta x} \cdot {y''}_{n+1}\\
\end{aligned}
$$

$$
{\Delta x} \cdot {y''}_{n+1} = \Delta y' \rightsquigarrow
$$

$$
\begin{aligned}
    y_{n+1} & = y_n + \Delta x \cdot {y'}_{n+1} +  {\Delta x} / 2 \cdot \Delta y' \\
    & = y_n + \Delta x \cdot ({y'}_{n+1} +  \Delta y / 2)'\\
    y'_{n+1} & = y'_n + \Delta y'\\
\end{aligned}
$$

$$
\rightsquigarrow
$$

$$
\begin{aligned}
    0 & = y_{n+1} - (y_n + \Delta x \cdot ({y'}_{n+1} +  \Delta y' / 2)) = gy\\
    0 & = y'_{n+1} - (y'_n + \Delta y') = gdy\\
\end{aligned}
$$

$$
\begin{aligned}
    J_{gy} & = \begin{bmatrix} I&0\end{bmatrix} -  {\Delta x} \cdot (\begin{bmatrix} 0&I\end{bmatrix} + J_{\Delta y'} / 2)\\
    J_{gdy} &= \begin{bmatrix} 0&I\end{bmatrix} - J_{\Delta y'}\\
\end{aligned}
$$

$$
\begin{aligned}
    \begin{bmatrix} I&0\end{bmatrix} &= I_y\\
    \begin{bmatrix} 0&I\end{bmatrix} &= I_{y'}\\
\end{aligned}
$$



$$
J_{\Delta y'} = \Delta x \cdot J(f)(y_{n+1}, {y'}_{n+1}) = xJf
$$

$$
\begin{aligned}
    J_{gy} & = I_y -  {\Delta x} \cdot (I_{y'} + xJf / 2)\\
    J_{gdy} &= I_{y'} - xJf\\
\end{aligned}
$$

Linear equation to be solved
$$
\begin{aligned}
    J_g \cdot a &= g\\
    u_{m+1} &= u_m - a
\end{aligned}
$$

Where

$$
J_g =
\begin{bmatrix}
    J_{gy} \\
    J_{gdy}\\
\end{bmatrix}
= I -
\begin{bmatrix}
    {\Delta x} \cdot (I_{y'} + xJf / 2) \\
    xJf\\
\end{bmatrix}
$$

$$
g =
\begin{bmatrix}
    gy \\
    gdy\\
\end{bmatrix}
$$
