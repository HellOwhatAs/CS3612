1. **请证明Lecture 3，Page 14末尾处介绍的Gaussain regression的beta的分布的均值恰好等价于Ridge regression的解——教材上没有，课堂上讲过。**
    要证明高斯回归 (Gaussian regression) 的 $\beta$ 分布的均值等于岭回归 (Ridge regression) 的解，需要首先定义一些符号和假设：
    假设我们有 $n$ 个训练样本，其中第 $i$ 个样本的特征向量为 $\mathbf{x}_i$，响应变量为 $y_i$。记 $\mathbf{X}$ 为 $n\times p$ 的设计矩阵，其中第 $i$ 行为 $\mathbf{x}_i^T$，$\mathbf{y}$ 为 $n\times 1$ 的响应向量，其中第 $i$ 个元素为 $y_i$。我们考虑一个线性回归模型，其中 $\beta$ 为 $p\times 1$ 的未知系数向量，模型可表示为：
    $$
    \mathbf{y} = \mathbf{X}\beta + \epsilon
    $$
    其中 $\epsilon$ 是 $n\times 1$ 的噪声向量，假设其服从均值为 $\mathbf{0}$，协方差矩阵为 $\sigma^2 \mathbf{I}$ 的多元高斯分布，即 $\epsilon \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$。
    岭回归是一种正则化的线性回归方法，它通过添加一个 $L_2$ 正则化项，使得优化问题变为：
    $$
    \hat{\beta}_\text{ridge} = \argmin_\beta\left\{\lVert \mathbf{y} - \mathbf{X}\beta \rVert^2 + \lambda \lVert \beta \rVert^2\right\}
    $$
    其中 $\lambda$ 是正则化超参数，控制了正则化的强度。可以证明，岭回归的解为：
    $$
    \hat{\beta}_\text{ridge} = \left(\mathbf{X}^\top\mathbf{X} + \lambda \mathbf{I}\right)^{-1} \mathbf{X}^\top \mathbf{y}
    $$
    接下来，我们考虑高斯回归模型的 $\beta$ 的分布。在高斯回归模型中，$\beta$ 的后验分布为：
    $$
    p(\beta|\mathbf{X}, \mathbf{y}) \propto p(\mathbf{y}|\mathbf{X}, \beta) p(\beta)
    $$
    其中 $p(\mathbf{y}|\mathbf{X},\beta)$ 是似然函数，即：
    $$
    p(\mathbf{y}|\mathbf{X}, \beta) = \frac{1}{\left(2\pi\sigma^2\right)^\frac{n}{2}}\exp\left(-\frac{1}{2\sigma^2}\lVert \mathbf{y} - \mathbf{X}\beta \rVert^2\right)
    $$
    $p(\beta)$ 是先验分布。我们假设 $\beta$ 服从均值为 $\mathbf{0}$，协方差为 $\tau^2 \mathbf{I}$ 的多元高斯分布，即 $\beta \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})$。

    将似然函数和先验分布代入后验分布中，得到：
    $$
    p(\beta|\mathbf{X}, \mathbf{y}) \propto \exp\left(-\frac{1}{2\sigma^2}\lVert \mathbf{y} - \mathbf{X}\beta \rVert^2 - \frac{1}{2r^2}\lVert\beta\rVert^2\right)
    $$
    我们可以将后验分布看作一个高斯分布的常数倍数，因此 $\beta$ 的后验分布也是一个多元高斯分布。具体地，后验分布可以表示为：
    $$
    \beta|\mathbf{X}, \mathbf{y} \sim \mathcal{N}\left(\mu_\text{post}, \Sigma_\text{post}\right)
    $$
    其中后验均值 $\mu_{\mathrm{post}}$ 和后验协方差矩阵 $\Sigma_{\mathrm{post}}$ 分别为：
    $$
    \mu_\text{post} = \frac{1}{\sigma^2}\left(\mathbf{X}^\top\mathbf{X} + \frac{r^2}{\sigma^2}\mathbf{I}\right)^{-1}\mathbf{X}^\top\mathbf{y}\\
    \Sigma_\text{post} = \left(\frac{1}{\sigma^2}\mathbf{X}^\top\mathbf{X}+\frac{1}{r^2}\mathbf{I}\right)^{-1}
    $$
    我们可以看出，后验均值 $\mu_{\mathrm{post}}$ 和岭回归的解形式相同，只是 $\lambda$ 被替换为了 $\frac{\tau^2}{\sigma^2}$。因此，当取 $\tau^2 = \lambda\sigma^2$ 时，高斯回归的后验均值 $\mu_{\mathrm{post}}$ 与岭回归的解 $\hat{\beta}_{\mathrm{ridge}}$ 是相等的。这个等价关系可以通过代入 $\tau^2 = \lambda\sigma^2$ 进行验证：
    $$
    \mu_\text{post} = \frac{1}{\sigma^2}\left(\mathbf{X}^\top\mathbf{X}+\frac{\lambda\sigma^2}{\sigma^2}\mathbf{I}\right)^{-1}\mathbf{X}^\top\mathbf{y} = \hat{\beta}_\text{ridge}
    $$
    因此，我们证明了高斯回归的 $\beta$ 分布的均值与岭回归的解是等价的。