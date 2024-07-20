# Second Order Solvers




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
