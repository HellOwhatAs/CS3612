1. **请写一下LSTM的基本公式，并介绍一下各个gates的意义。**
   $$\begin{aligned}
       & \Delta c_t = f\left(W_c\left(h_{t-1}, x_t\right)\right)\\
       & i_t = f\left(W_i\left(h_{t-1}, x_t\right)\right)\\
       & f_t = f\left(W_f\left(h_{t-1}, x_t\right)\right)\\
       & c_t = c_{t-1}f_t + \Delta c_t i_t\\
       & o_t = f\left(W_o\left(h_{t-1}, x_t\right)\right)\\
       & h_t = o_t f\left(c_t\right)\\
       & y_t = f\left(Wh_t\right)
   \end{aligned}$$
   
   - input gate $i_t$：控制输入信息的进入，决定在当前时刻有哪些输入信息会被加入到状态中。
   - forget gate $f_t$：控制过去的信息的遗忘，决定哪些过去的信息应该被忘记。
   - output gate $o_t$：控制输出信息的提取，决定当前时刻哪些状态信息应该被输出。

2. **请写一下GAN模型的Loss function V(G,D)，以及其训练的min max 形式。解释一下Loss的意义。**
   $$\begin{aligned}
       V\left(D, G\right) &= \text{E}_{P_\text{data}}\left[\log p_D(X)\right] + \text{E}_{h\sim p(h)}\left[\log\left(1-p_D\left(G(h)\right)\right)\right]\\
       &= \text{E}_{P_\text{data}}\left[\log D(X)\right] + \text{E}_{h\sim p(h)}\left[\log\left(1-D(G(h))\right)\right]
   \end{aligned}$$
   
   训练的min max 形式:
   $$\min_G\max_DV(D, G)$$
   
   该 Loss 的意义是：第一项是真实数据的对数概率，表示判别器对真实数据的判别结果；第二项是生成器生成的假数据的对数概率，表示判别器对假数据的判别结果。

3. **请证明优化GAN模型等价于优化JSD(P_data || P_theta)。介绍一下P_theta的公式和意义——这个课堂上讲过。**
   $$p_\text{mix} = \frac{P_\text{data} + p_\theta}{2}$$
   $$\begin{aligned}
       \text{JSD}\left(P_\text{data}|p_\theta\right) &= \text{KL}\left(p_\theta|p_\text{mix}\right) + \text{KL}\left(P_\text{data}|p_\text{mix}\right)\\
       &= \sum_X\left[p_\theta(X)\log\frac{p_\theta(X)}{p_\text{mix}(X)} + P_\text{data}(X)\log \frac{P_\text{data}(X)}{p_\text{mix}(X)}\right]\\
       &= - H(p_\theta) - H(P_\text{data}) - \sum_X\left[p_\theta(X)\log p_\text{mix}(X) + P_\text{data}(X)\log p_\text{mix}(X)\right]\\
       &= -H(p_\theta)-H(P_\text{data})-\sum_X\left[p_\theta(X)\log\frac{p_\theta(X)}{2(1-D(X))} + P_\text{data}(X)\log\frac{P_\text{data}(X)}{2D(X)}\right]\\
       &= -H(p_\theta)-H(P_\text{data})+H(p_\theta)+H(P_\text{data}) + \sum_{X}\left[p_\theta(X)\log\left[2(1-D(X))\right] + P_\text{data}(X)\log(2D(X))\right]\\
       &= \sum_X\left[p_\theta(X)\log2 + P_\text{data}(X)\log2\right] + \sum_X\left[p_\theta(X)\log(1-D(X)) + P_\text{data}(X)\log(D(X))\right]\\
       &= 2\log2 + V(D, G)
   \end{aligned}$$
   因此，优化GAN模型等价于优化 $\text{JSD}\left(P_\text{data}|p_\theta\right)$。
   
   $P_\theta(X)$ 的公式：
   $$P_\theta(X) = \sum_hP(h)P_G(X|h)$$
   $P_\theta(X)$ 表示 $X$ 被生成出来的概率。