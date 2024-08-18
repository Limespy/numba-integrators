# Second Order Solvers



## RKNF

order = $o$
steps = $s$

### Amount of parameters

#### Array sizes

In base formulation

- One array $\alpha$
  -  $s + 1$
- Two triangular matrices $\beta$ and $\gamma$
  - $(s+1, s)$

#### Initial costraints

$\beta$ and $\beta$ are padded
$$
\begin{matrix}
    \alpha[0] & = & 0\\
    \beta[0] & = & [0, ...]\\
    \gamma[0] & = & [0, ...]\\
\end{matrix}
$$

> 4 constraints

$$
\begin{matrix}
    \alpha[-2] & = & 1\\
    \alpha[-1] & = & 1\\
    \beta[1, 0] & = & \alpha[1]\\
    \gamma[1, 0] & = & 0.5 \cdot \alpha[1]^2\\
\end{matrix}
$$

#### Total

So in total
$$
n_{parameters} = s + 2 \cdot (s \cdot (s + 1)/2) - 4
= s + 1 + s \cdot (s + 1) - 5
= (s + 1)^2 - 5
$$

|   s   | params |
| :---: | :----: |
|   6   |   44   |
|   7   |   59   |
|   8   |   76   |
|   9   |   95   |
|  10   |  116   |
|  11   |  139   |

### Equations

---

> $ 2 \cdot (2 \cdot (s - 1)) = 4 \cdot s - 4$ equations

For p in {0, 1, ..., o-3}


$$
\bm{\beta}[2:] \cdot \bm{a}[:-1]^{\circ p}
 = \frac{\bm{\alpha}[2:]^{\circ p + 1}}{p+1}
$$
$$
\bm{\gamma}[2:] \cdot \bm{a}[:-1]^{\circ p}
 = \frac{\bm{\alpha}[2:]^{\circ p + 2}}{(p+1) \cdot (p+2)}
$$

---


> $2 \cdot (2 \cdot o - 1  + 5) = 4 \cdot o + 8$ equations

For $i$ in {s-1, s}

For $p$ in {0, 1, 2, ..., o - 1 + $i$ - $s$}

$$
\bm{\beta}[i] \cdot \bm{a}^{\circ p} = 1 / (p+1)
$$

$$
\bm{\gamma}[i] \cdot \bm{a}^{\circ p} = 1 / ((p+1) \cdot (p+2))
$$


$$
(\gamma[:-1] \cdot \alpha[:-1]) \cdot \beta[i] = 1/120
$$

$$
(\beta[:-1] \cdot \alpha[:-1]) \cdot \beta[i] = 1/24
$$

$$
(\beta[:-1] \cdot \alpha[:-1]^{\circ 2}) \cdot \beta[i] = 1/60
$$

$$
(\beta[:-1] \cdot \alpha[:-1]) \cdot (\beta[i] \odot \alpha[:-1]) = 1/40
$$

$$
(\beta[:-1] \cdot (\beta[:-1] \cdot \alpha[:-1])) \cdot \beta[i] = 1/120
$$


---



#### Total

$ 4 \cdot o + 8 + 4 \cdot s - 4 = 4 \cdot (s + o + 1) $

Equations

|  s\o  |   4   |   5   |   6   |   7   |
| :---: | :---: | :---: | :---: | :---: |
|   6   |  32   |  36   |  40   |  44   |
|   7   |  36   |  40   |  44   |  48   |
|   8   |  40   |  44   |  48   |  52   |
|   9   |  44   |  48   |  52   |  56   |
|  10   |  48   |  52   |  56   |  60   |
|  11   |  52   |  56   |  60   |  64   |

Free parameters

|  s\o  |   4   |   5   |   6   |   7   |
| :---: | :---: | :---: | :---: | :---: |
|   6   | *12*  |   8   |   4   |   0   |
|   7   |  23   |  19   |  15   |  11   |
|   8   |  36   | *32*  |  28   |  24   |
|   9   |  51   |  47   |  43   |  39   |
|  10   |  68   |  64   |  60   |  56   |
|  11   |  87   |  83   |  79   |  75   |

### Solving



#### Inputs in RKF56

$$
\begin{matrix}
\alpha[4] & = & 9/10\\
\alpha[5] & = & 3/4\\
\alpha[6] & = & 2/4\\
\end{matrix}
$$
3

$$
\begin{matrix}
\beta[3,1] & = & 0\\
\beta[4,1] & = & 0\\
\beta[4,2] & = & 0\\
\beta[5,1] & = & 0\\
\beta[5,2] & = & 0\\
\beta[6,1] & = & 0\\
\beta[6,2] & = & 0\\
\beta[7,1] & = & 0\\
\beta[7,2] & = & 0\\
\beta[8,1] & = & 0\\
\beta[8,2] & = & 0\\
\beta[8,3] & = & 0\\
\end{matrix}
$$
12

$$
\begin{matrix}
\gamma[4,1] & = & 0\\
\gamma[5,1] & = & 0\\
\gamma[6,1] & = & 0\\
\gamma[6,2] & = & 0\\
\gamma[6,3] & = & 0\\
\gamma[7,1] & = & 0\\
\gamma[7,2] & = & 0\\
\gamma[7,3] & = & 0\\
\gamma[8,1] & = & 0\\
\gamma[8,2] & = & 0\\
\gamma[8,3] & = & 0\\
\end{matrix}
$$
11

total 26
## Other

### Approximator


$$
x = -\Delta x
$$

$$
\begin{matrix}
A(0) & = & y_0\\
A'(0) & = & y'_0\\
A''(0) & = & y''_0\\
A(x) & = & y_{-1}\\
A'(x) & = & y'_{-1}\\
A''(x) & = & y''_{-1}\\
\end{matrix}
$$

#### Poly5

$$
\begin{matrix}
A(x) & = & p_0 + p_1 \cdot x + p_2 \cdot x^2 + p_3 \cdot x^3 + p_4 \cdot x^4
    + p_5 \cdot x^5\\
A'(x) & = & p_1 + p_2 \cdot 2 \cdot x + p_3 \cdot 3 \cdot x^2
    + p_4 \cdot 4 \cdot x^3 + p_5 \cdot 5 \cdot x^4\\
A''(x) & = & p_2 \cdot 2 + p_3 \cdot 6 \cdot x
    + p_4 \cdot 12 \cdot x^2 + p_5 \cdot 20 \cdot x^3\\
\end{matrix}
$$
First parameters easily
$$
\begin{matrix}
p_0 & = & y_0\\
p_1 & = & y'_0\\
p_2 & = & y''_0 / 2\\
\end{matrix}
$$

Next three more complicated
$$
\begin{matrix}
y_{-1} & = & p_0 + p_1 \cdot x + p_2 \cdot x^2 + p_3 \cdot x^3 + p_4 \cdot x^4
    + p_5 \cdot x^5\\
y'_{-1} & = & p_1 + p_2 \cdot 2 \cdot x + p_3 \cdot 3 \cdot x^2
    + p_4 \cdot 4 \cdot x^3 + p_5 \cdot 5 \cdot x^4\\
y''_{-1} & = & p_2 \cdot 2 + p_3 \cdot 6 \cdot x
    + p_4 \cdot 12 \cdot x^2 + p_5 \cdot 20 \cdot x^3\\
\end{matrix}
$$

$$
\begin{matrix}
y_{-1} - (p_0 + p_1 \cdot x + p_2 \cdot x^2) & = & p_3 \cdot x^3 + p_4 \cdot x^4
    + p_5 \cdot x^5\\
y'_{-1} - (p_1 + p_2 \cdot 2 \cdot x)& = & p_3 \cdot 3 \cdot x^2
    + p_4 \cdot 4 \cdot x^3 + p_5 \cdot 5 \cdot x^4\\
y''_{-1} - (p_2 \cdot 2) & = & p_3 \cdot 6 \cdot x
    + p_4 \cdot 12 \cdot x^2 + p_5 \cdot 20 \cdot x^3\\
\end{matrix}
$$

$$
\begin{matrix}
y_{-1} - (p_0 + p_1 \cdot x + p_2 \cdot x^2) & = & p_3 \cdot x^3 + p_4 \cdot x^4
    + p_5 \cdot x^5\\
y'_{-1} - (p_1 + y''_0 \cdot x)& = & p_3 \cdot 3 \cdot x^2
    + p_4 \cdot 4 \cdot x^3 + p_5 \cdot 5 \cdot x^4\\
y''_{-1} - y''_0 & = & p_3 \cdot 6 \cdot x
    + p_4 \cdot 12 \cdot x^2 + p_5 \cdot 20 \cdot x^3\\
\end{matrix}
$$
