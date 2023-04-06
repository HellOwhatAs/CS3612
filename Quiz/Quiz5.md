1. **请默写出拉格朗日乘子法的一般形式。见Page 6中，同时包含等式条件和不等式条件下的拉格朗日乘子法的问题所对应的一般形式，和解法的一般形式。（不是写对SVM的求解过程，只写一般形式，而且写出解法的一般套路即可，“对L求梯度=0”这一步为止）**
$$
\min_w f(w)\\
\begin{aligned}
    \text{s.t.} \qquad &g_k(w) \le 0,\quad k = 1, 2,\cdots,K\\
    &h_l(w) = 0,\quad l =1, 2,\cdots, L
\end{aligned}
$$
根据拉格朗日乘子法，我们可以构造拉格朗日函数
$$
L(w, \alpha, \beta) = f(w) +\sum_{k=1}^K\alpha_kg_k(w) +\sum_{l=1}^L
\beta_lh_l(w) \\
\min_w\max_{\alpha, \beta: \alpha_k\ge 0}L(w,\alpha,\beta)
$$
KKT条件如下：
$$
\begin{aligned}
    \forall p, \frac{\partial}{\partial w_p}L(w, \alpha, \beta) &= 0\\
    \forall l, \frac{\partial}{\partial \beta_l}L(w, \alpha, \beta) &= 0\\
    \forall k, \alpha_kg_k(w) &= 0\\
    \forall k, g_k(w) &\le 0\\
    \forall k, \alpha_k &\ge 0
\end{aligned}
$$

1. **Page 8：仅考虑当“数据线性可分”的情况下，从拉格朗日乘子法的角度，解释为什么SVM的参数w的方向仅决定于数据中的支持向量。此问题作答包括三个方面。**
**a. 什么叫“支持向量”。**
**b. 如何推导出w的解的形式。**
**c. 从拉格朗日乘子法的角度解释一下（不一定严谨的证明），为什么仅有支持向量所对应的alpha_i才可能大于0。或者解释为什么不是支持向量的样本的alpha_i必须等于0。**
**综合上面三个方面，可以解释为什么SVM的参数w的方向仅决定于数据中的支持向量。**
   - a. 支持向量是 $\alpha_i$ 大于0，从而保证满足 $−y_i(w>X_i +b)+1 = 0$ 条件的训练样本点。
   - b. 在线性可分的情况下，SVM的优化问题可以表示为：
最小化目标函数：
min ||w||^2
满足约束条件：
yi(w.Txi + b) - 1 ≥ 0, i = 1, ..., n
其中，||w||表示向量w的模长，b表示超平面的截距，xi和yi分别表示第i个样本的特征和标签。
根据拉格朗日乘子法，我们可以构造拉格朗日函数：
L(w, b, α) = 1/2||w||^2 - ∑αi[yi(w.Txi + b) - 1]
其中，αi是拉格朗日乘子，用于表示约束条件的重要性程度。
接下来，我们可以通过求解L(w, b, α)的梯度为零的方程组来得到w和b的解。具体而言，我们可以先对w求偏导数，得到w的表达式：
w = ∑αiyixi
这个表达式表明，w的方向和大小都是由支持向量的权重和xi决定的，而与非支持向量无关。
c. 从拉格朗日乘子法的角度解释，我们知道在SVM中，只有支持向量所对应的αi才可能大于0，因为对于非支持向量的样本，它们的αi必须等于0，否则它们不满足约束条件，优化问题就不能得到正确的解。由于w的表达式中只包含了αi乘以yi和xi的线性组合，而非支持向量的αi为0，因此它们在w的表达式中不起任何作用。因此，w的方向和大小只与支持向量的权重和xi有关，而与非支持向量无关，这就是SVM的参数w的方向仅决定于数据中的支持向量的原因。
1. **Page13：请默写出带outlier时，SVM的目标函数的定义（带不等式条件的定义），以及SVM的等效目标函数的定义（不带不等式条件的定义），以及SVM在拉格朗日乘子法下的损失函数。（只写定义，不写求解过程）**
$$
\min_{\xi, w, b} \frac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^n\xi_i\\
\begin{aligned}
    \text{s.t.} \qquad &y_i\left(w^\top X_i + b\right)\ge 1-\xi_i, \quad i = 1,2,\cdots,n\\
    &\xi_i\ge0, \quad i=1,2,\cdots,n
\end{aligned}
$$
$$
\min_{\xi, w, b} \frac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^n\max\left(0, 1- y_i\left(w^\top X_i+b\right)\right)
$$
$$
L(w,b,\xi,\alpha,\beta) = \frac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^n\xi_i-\sum_{i=1}^n\alpha_i\left[y_i\left(w^\top X_i+b\right) - 1+\xi_i\right] - \sum_{i=1}^n\beta_i\xi_i\\
\min_{w,b,\xi:\xi_i\ge 0}\max_{\alpha,\beta:\alpha_i\ge 0,\beta_i\ge 0}L(w,b,\xi,\alpha,\beta)
$$