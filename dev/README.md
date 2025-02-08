# Implicit solvers, second order

## Solving

### Backwards differentiation

$$
y'' = f(x, y, y')
$$

$$
\begin{aligned}
    y_{n+1} & = y_n + {y'}_{n+1} \cdot \Delta x +  {y''}_{n+1} / 2  \cdot \Delta x\\
    y'_{n+1} & = y'_n + {y''}_{n+1} \cdot \Delta x\\
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

### Central difference

$$
{y''}_{n} = f(x_{n}, y_{n}, {y'}_{n})
$$

#### Solving $y_{n+1}$ and ${y'}_{n+1}$

Using two-point taylor polynomial

$$
\begin{aligned}
    y_n             & = P(\Delta x)
    P(0)            & = {y}_{n}                         \\
    P'(0)           & = {y'}_{n}                        \\
    P''(0)          & = {y''}_{n}                       \\
    P''(\Delta x)   & = {y''}_{n + 1}
\end{aligned}
$$

$$
\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 2 & 0\\
\end{bmatrix} \cdot
\begin{bmatrix}
p_0\\
p_1\\
p_2\\
p_3
\end{bmatrix} =
\begin{bmatrix}
P(0)\\
P'(0)\\
P''(0)\\
\end{bmatrix} \rightsquigarrow
\begin{bmatrix}
p_0\\
p_1\\
p_2\\
\end{bmatrix} =
\begin{bmatrix}
P(0)\\
P'(0)\\
P''(0)/2\\
\end{bmatrix}
$$

$$
\begin{bmatrix}
0 & 0 & 2 & 6\\
\end{bmatrix} \cdot
\begin{bmatrix}
p_0\\
p_1\\
p_2\\
p_3 \cdot {\Delta x}\\
\end{bmatrix} =
\begin{bmatrix}
P''(\Delta x)\\
\end{bmatrix}
$$

Combining those two

$$
\begin{aligned}
p_3 & = (P''(\Delta x) - 2 \cdot p_2) / (6 \cdot {\Delta x})\\
& = (P''(\Delta x) - P''(0)) / (6 \cdot {\Delta x})
\end{aligned}
$$

$$
\begin{aligned}
    y(\Delta x) & = p_0 + p_1 \cdot {\Delta x} + p_2 \cdot {\Delta x}^2 + p_3 \cdot {\Delta x}^3\\
    y'(\Delta x) & = p_1 + p_2 \cdot 2 \cdot \Delta x + p_3 \cdot 3 \cdot {\Delta x}^2
\end{aligned}
$$

Substituting the known P(x)

$$
\begin{aligned}
    y(\Delta x) & = P(0)
                    + P'(0) \cdot {\Delta x}
                    + P''(0) / 2 \cdot {\Delta x}^2
                    + (P''(\Delta x) - P''(0)) / 6  \cdot {\Delta x}^2\\
    y'(\Delta x) & = P'(0) + P''(0) \cdot \Delta x + (P''(\Delta x) - P''(0)) / 2 \cdot {\Delta x}
\end{aligned}
$$

$$
\begin{aligned}
    y(\Delta x) & = P(0)
                    + P'(0) \cdot {\Delta x}
                    + (P''(0) / 3 + P''(\Delta x) / 6) \cdot {\Delta x}^2\\
    y'(\Delta x) & = P'(0) + (P''(0) + P''(\Delta x)) / 2 \cdot {\Delta x}
\end{aligned}
$$

$$
\begin{aligned}
    y_{n+1} & = y_n
                + {y'}_n \cdot {\Delta x}
                + ({y''}_n / 3 + {y''}_{n+1} / 6) \cdot {\Delta x}^2\\
    {y'}_{n+1} & = {y'}_n + ({y''}_n + {y''}_{n+1}) / 2 \cdot {\Delta x}
\end{aligned}
$$

#### Equations for solver

$$
\begin{aligned}
    gy({y}_{n+1}, {y'}_{n+1}) & =
        y_{n+1}- (y_n
                  + {y'}_n \cdot {\Delta x}
                  + ({y''}_n / 3 + {y''}_{n+1} / 6) \cdot {\Delta x}^2)\\
    gdy({y}_{n+1}, {y'}_{n+1}) & =
        {y'}_{n+1} - ({y'}_n + ({y''}_n + {y''}_{n+1}) / 2 \cdot {\Delta x})
\end{aligned}
$$

$$
\begin{aligned}
    gy({y}_{n+1}, {y'}_{n+1}) & =
        y_{n+1}- (y_n
                  + {y'}_n \cdot {\Delta x}
                  + ({y''}_n / 3 + {y''}_{n+1} / 6) \cdot {\Delta x}^2)\\
    gdy({y}_{n+1}, {y'}_{n+1}) & =
        {y'}_{n+1} - ({y'}_n + ({y''}_n + {y''}_{n+1}) / 2 \cdot {\Delta x})
\end{aligned}
$$

$$
\begin{aligned}
    gy({y}_{n+1}, {y'}_{n+1}) & =
        y_{n+1} - (y_n
                  + {y'}_n \cdot {\Delta x}
                  + ({y''}_n / 3 + {y''}_{n+1} / 6) \cdot {\Delta x}^2)\\
    gdy({y}_{n+1}, {y'}_{n+1}) & =
        {y'}_{n+1} - ({y'}_n + ({y''}_n + {y''}_{n+1}) / 2 \cdot {\Delta x})
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
    J(gy) \\
    J(gdy)\\
\end{bmatrix} =
\begin{bmatrix}
    J_{gy} \\
    J_{gdy}\\
\end{bmatrix}
$$


$$
\begin{aligned}
    gy({y}_{n+1}, {y'}_{n+1}) & =
        y_{n+1}- (y_n
                  + {y'}_n \cdot {\Delta x}
                  + ({y''}_n / 3 + {y''}_{n+1} / 6) \cdot {\Delta x}^2)\\
    J_{gdy} & = I_{y'} -  J_{y''_{n+1}} / 2 \cdot {\Delta x}
\end{aligned}
$$

$$
g =
\begin{bmatrix}
    gy \\
    gdy\\
\end{bmatrix}
$$

#### More general

$$
h(j, k) = \Pi_{i=0}^{j - 1}(k - i)
$$

$$
\begin{aligned}
    NM & = \R^{d,2 \cdot d - 1} = \R^{i,j}\\
    & =
    \begin{bmatrix}
    1      & 1      & 1      & \dots  & 1                 & 1 \\
    0      & 1      & 2      & \dots  & j - 2             & j - 1 \\
    0      & 0      & 2      & \dots  & h(2, j-2) = (j-2) \cdot (j-3) & h(2, j-1) = (j-1) \cdot (j-2) \\
    \vdots & \vdots & \vdots & \ddots & \vdots            & \vdots\\
    0      & 0      & 0      & \dots  & h(i-2, j-1) & h(i-1, j-1) \\
    \end{bmatrix}\\
    & = \begin{bmatrix}
    1 = nm_{1,1}      & 1      & 1      & \dots  & 1                 & 1 \\
    0      & 1      & 2      & \dots  & nm_{1, j - 1} \cdot (j - 2) & nm_{1, j} \cdot (j - 1) \\
    0      & 0      & 2      & \dots  & nm_{2, j - 1} \cdot (j - 3) & nm_{2, j} \cdot (j - 2) \\
    \vdots & \vdots & \vdots & \ddots & \vdots            & \vdots\\
    0      & 0      & 0      & \dots  & nm_{i-1, j - 1} \cdot (j - i) & nm_{n-1, j} \cdot (j - (j-1)) \\
    \end{bmatrix}\\
    & = \begin{bmatrix}N & M\end{bmatrix}\\
\end{aligned}
$$

$$
\begin{aligned}
N &= NM[:, :d]\\
M &= NM[:, d:]
\end{aligned}
$$

$$
\begin{aligned}
F &= \R^{d,d}\\
&=
    \begin{bmatrix}
    0! & 0 & \dots & 0\\
    0 & 1! & \dots & 0\\
    \vdots & \vdots & \ddots & \vdots\\
    0 & 0 & \dots & (d-1)!
    \end{bmatrix}\\
RF &= \R^{d,d}\\
&=
    \begin{bmatrix}
    1/0! & 0 & \dots & 0\\
    0 & 1/1! & \dots & 0\\
    \vdots & \vdots & \ddots & \vdots\\
    0 & 0 & \dots & 1/(d-1)!
    \end{bmatrix}
\end{aligned}
$$

$$
\begin{aligned}
DA &= \R^{d,1}\\
&=
    \begin{bmatrix}
    f(x_a)\\ D_x^1(f)(x_a)\\ \vdots \\ D_x^{d-1}(f)(x_a)
    \end{bmatrix}\\
DB &= \R^{d,1}\\
&=
    \begin{bmatrix}
    f(x_b)\\ D_x^1(f)(x_b)\\ \vdots \\ D_x^{d-1}(f)(x_b)
    \end{bmatrix}\\
\end{aligned}
$$

$$
\begin{aligned}
X &= \R^{2 \cdot d - 1,2 \cdot d - 1}\\
&=
    \begin{bmatrix}
        1 & 0 & 0 & \dots & 0\\
        0 & x & 0 & \dots & 0\\
        0 & 0 & x^2 & \dots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \dots & x^{2\cdot d - 2}\\
    \end{bmatrix}\\
\end{aligned}
$$

$$
\begin{aligned}
X_{:d, :d} \cdot DB
    &= NM \cdot X \cdot P\\
DB
    &= X_{:d, :d}^{-1} \cdot (N \cdot X_{:d,:d} \cdot PA
                              + M \cdot X_{d:,d:} \cdot PB)
\end{aligned}
$$

$$
\begin{aligned}
F \cdot PA &= DA\\
PA &= F^{-1} \cdot DA\\
F^{-1} &= IF\\
PA &= IF \cdot DA
\end{aligned}
$$

$$
\begin{aligned}
X_{:d!v,:d} \cdot DB_{!v}
    &= NM_{!v,:} \cdot X \cdot P\\
    &= N_{!v,:} \cdot X_{:d,:d} \cdot PA
       + M_{!v,:} \cdot X_{d:!v,d:} \cdot PB \\
M_{!v,:} \cdot X_{d:!v,d:} \cdot PB
    &= X_{:d!v,:d} \cdot DB_{!v}
       - N_{!v,:} \cdot X_{:d,:d} \cdot PA \\
X_{d:!v,d:} \cdot PB
    &= M_{!v,:}^{-1} \cdot (X_{:d!v,:d} \cdot DB_{!v}
                            - N_{!v,:} \cdot X_{:d,:d} \cdot PA) \\
M_{!v,:}^{-1}
    &= IM\\
PB
    &=  X_{d:!v,d:}^{-1} \cdot IM \cdot (X_{:d!v,:d} \cdot DB_{!v}
                                       - N_{!v,:} \cdot X_{:d,:d} \cdot PA) \\
X_{:d!v,d:}^{-1}
    &= IX\\
PB &= IX \cdot IM \cdot (X_{:d!v,:d} \cdot DB_{!v}
                         - N_{!v,:} \cdot X_{:d,:d} \cdot PA) \\
\end{aligned}
$$

Combining

$$
\begin{aligned}
DB
    &= X_{:d, :d}^{-1} \cdot (N \cdot X_{:d,:d} \cdot PA
                              + M \cdot X_{d:,d:} \cdot PB)
X_{:d, :d}^{-1} \cdot N \cdot X_{:d,:d}
    &= H
X_{:d, :d}^{-1} \cdot M \cdot X_{d:,d:}
    &= J

DB
    &= H \cdot PA + J \cdot PB
    &= H \cdot PA + J \cdot IX \cdot IM \cdot (X_{:d!v,:d} \cdot DB_{!v}
                         - N_{!v,:} \cdot X_{:d,:d} \cdot PA)
J \cdot IX \cdot IM \cdot X_{:d!v,:d}
    &= K
J \cdot IX \cdot IM \cdot N_{!v,:} \cdot X_{:d,:d}
    &= L
DB
    &= (H - L ) \cdot PA + L \cdot DB_{!v}
    &= (H - L ) \cdot IF \cdot DA + L \cdot DB_{!v}
(H - L ) \cdot IF
    &= V
DB
    &= V \cdot DA + K \cdot DB_{!v}
\end{aligned}
$$


$$
a = 1
$$

<!--
$$
\begin{aligned}
    N & = \R^{d,d} = \R^{i,j}\\
    & =
    \begin{bmatrix}
    1 = n_{1,1}     & 1      & \dots  & 1                \\
    0      & 1      & \dots  & n_{1, j-1} \cdot (j - 2) \\
    \vdots & \vdots & \ddots & \vdots \\
    0      & 0      & \dots  & n_{i-1, j-1} \cdot (j - (i-1)) \\
    \end{bmatrix}\\
    M & = \R^{d,d-1} = \R^{i,j}\\
    & =
    \begin{bmatrix}
    1 = n_{1,1}     & 1      & \dots  & 1                \\
    m_{1, 1} \cdot (d - (2 - 2)) & m_{1, 2} \cdot (d + 1) & \dots  & m_{1, j-1} \cdot (j - 2) \\
    \vdots & \vdots & \ddots & \vdots \\
    m_{i - 1, 1} \cdot (d - i -) & 0      & \dots  & n_{i-1, j-1} \cdot (j - (i-1)) \\
    \end{bmatrix}\\
\end{aligned}
$$ -->


$$
D_x^n(y) = N()
\begin{bmatrix}
p_0\\p_1 \cdot \Delta x\\p_2 \cdot {\Delta x}^2\\\vdots\\p_n \cdot {\Delta x}^n
\end{bmatrix}
$$


$$
P\Delta X =
\begin{bmatrix}
p_0\\p_1 \cdot \Delta x\\p_2 \cdot {\Delta x}^2\\\vdots\\p_n \cdot {\Delta x}^n
\end{bmatrix}
$$

$$
P\Delta X =
\begin{bmatrix}
p_0\\p_1 \cdot \Delta x\\p_2 \cdot {\Delta x}^2\\\vdots\\p_n \cdot {\Delta x}^n
\end{bmatrix}
$$


### Padé approximant

$$
\begin{aligned}
R(x)
    &= \frac{\sum_{j=0}^{m}a_j \cdot x^j}{1 + \sum_{k=1}^{n}b_k \cdot x^k}
\sum_{j=0}^{m}a_j \cdot x^j
    &= g
\sum_{k=1}^{n}b_k \cdot x^k
    &= h
R(x)
    &= \frac{g}{h}
\end{aligned}
$$

$$
\begin{aligned}
R(x)
    &= \frac{g}{h}\\
D_x^1(R)(x)
    &= \frac{g' \cdot h - g \cdot h'}{h^2}\\
D_x^2(R)(x)
    &= \frac{h^2 \cdot g''
             - h \cdot (2 \cdot g' \cdot h' + g \cdot h'')
             + 2 \cdot g \cdot {h'}^2)}{h^3}\\
\end{aligned}
$$
